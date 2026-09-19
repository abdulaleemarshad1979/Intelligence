"""Tamper-Evident Cryptographic Audit Logger for Police Surveillance & Biometrics.

Implements an append-only cryptographic hash chain (SHA-256) logging every biometric query,
dossier view, match verification, video export, and policy change.
"""

import sqlite3
import hashlib
import json
import time
import uuid
import threading
from typing import Dict, Any, List, Optional, Tuple
from app.compliance.rbac import OfficerIdentity, DEFAULT_OFFICER

GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"
_LEDGER_LOCK = threading.Lock()

class AuditLogger:
    """Manages cryptographic tamper-evident audit logs."""

    def __init__(self, db_path: str = "data/police_records.db"):
        self.db_path = db_path
        self._init_audit_table()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_audit_table(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cryptographic_audit_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id TEXT UNIQUE NOT NULL,
                    timestamp REAL NOT NULL,
                    user_id TEXT NOT NULL,
                    badge_number TEXT NOT NULL,
                    officer_name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON cryptographic_audit_ledger(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_action ON cryptographic_audit_ledger(action_type)")
            conn.commit()

    def _get_latest_hash(self) -> str:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT entry_hash FROM cryptographic_audit_ledger ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                return row["entry_hash"]
            return GENESIS_HASH

    def compute_hash(
        self,
        prev_hash: str,
        timestamp: float,
        user_id: str,
        action_type: str,
        resource_id: str,
        details_json: str
    ) -> str:
        payload = f"{prev_hash}|{timestamp:.4f}|{user_id}|{action_type}|{resource_id}|{details_json}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def log_action(
        self,
        action_type: str,
        resource_id: str,
        details: Dict[str, Any],
        officer: Optional[OfficerIdentity] = None
    ) -> Dict[str, Any]:
        """Record an immutable, cryptographically chained audit log entry."""
        if officer is None:
            officer = DEFAULT_OFFICER

        with _LEDGER_LOCK:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT entry_hash FROM cryptographic_audit_ledger ORDER BY id DESC LIMIT 1")
                row = cursor.fetchone()
                prev_hash = row["entry_hash"] if row else GENESIS_HASH

                timestamp = time.time()
                entry_id = f"AUD-{int(timestamp*1000)}-{uuid.uuid4().hex[:6]}"
                details_json = json.dumps(details, sort_keys=True)
                entry_hash = self.compute_hash(prev_hash, timestamp, officer.user_id, action_type, resource_id, details_json)

                conn.execute("""
                    INSERT INTO cryptographic_audit_ledger (
                        entry_id, timestamp, user_id, badge_number, officer_name,
                        role, action_type, resource_id, details_json, prev_hash, entry_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry_id, timestamp, officer.user_id, officer.badge_number,
                    officer.full_name, officer.role.value, action_type,
                    resource_id, details_json, prev_hash, entry_hash
                ))
                conn.commit()

            return {
                "entry_id": entry_id,
                "timestamp": timestamp,
                "action_type": action_type,
                "resource_id": resource_id,
                "officer": officer.full_name,
                "badge_number": officer.badge_number,
                "entry_hash": entry_hash,
                "prev_hash": prev_hash
            }

    def get_logs(self, limit: int = 50, action_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve recent audit logs in reverse chronological order."""
        with self._get_connection() as conn:
            if action_filter:
                cursor = conn.execute(
                    "SELECT * FROM cryptographic_audit_ledger WHERE action_type = ? ORDER BY id DESC LIMIT ?",
                    (action_filter, limit)
                )
            else:
                cursor = conn.execute("SELECT * FROM cryptographic_audit_ledger ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            logs = []
            for r in rows:
                logs.append({
                    "entry_id": r["entry_id"],
                    "timestamp": r["timestamp"],
                    "formatted_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(r["timestamp"])),
                    "user_id": r["user_id"],
                    "badge_number": r["badge_number"],
                    "officer_name": r["officer_name"],
                    "role": r["role"],
                    "action_type": r["action_type"],
                    "resource_id": r["resource_id"],
                    "details": json.loads(r["details_json"]),
                    "prev_hash": r["prev_hash"],
                    "entry_hash": r["entry_hash"]
                })
            return logs

    def verify_integrity(self) -> Dict[str, Any]:
        """Traverse the cryptographic blockchain-style hash chain to certify database tamper-evidence."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM cryptographic_audit_ledger ORDER BY id ASC")
            rows = cursor.fetchall()

        if not rows:
            return {"verified": True, "total_records": 0, "status": "EMPTY_CHAIN"}

        prev = GENESIS_HASH
        for idx, r in enumerate(rows):
            if r["prev_hash"] != prev:
                return {
                    "verified": False,
                    "failed_at_entry": r["entry_id"],
                    "failed_at_index": idx,
                    "reason": f"Hash link broken. Expected prev_hash {prev[:10]}..., found {r['prev_hash'][:10]}..."
                }
            expected_hash = self.compute_hash(
                prev, r["timestamp"], r["user_id"], r["action_type"], r["resource_id"], r["details_json"]
            )
            if r["entry_hash"] != expected_hash:
                return {
                    "verified": False,
                    "failed_at_entry": r["entry_id"],
                    "failed_at_index": idx,
                    "reason": f"Content tampered! Recalculated hash does not match stored hash."
                }
            prev = r["entry_hash"]

        return {
            "verified": True,
            "total_records": len(rows),
            "status": "CHAIN_INTEGRITY_VERIFIED",
            "latest_hash": prev
        }
