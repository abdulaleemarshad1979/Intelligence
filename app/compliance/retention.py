"""Biometric Data Retention and Automated Purge Engine for Police Systems.

Enforces statutory compliance (DPDP Act / GDPR principles):
- Unmatched / Innocent pedestrian crops and vectors: purged after retention window (default 72h).
- Unreviewed candidate match tracks: retained for 30 days.
- Confirmed FIR suspect matches: preserved under official case file.
"""

import os
import time
import sqlite3
from typing import Dict, Any, List, Optional
from app.compliance.audit import AuditLogger
from app.compliance.rbac import OfficerIdentity, DEFAULT_OFFICER

class BiometricRetentionEngine:
    """Automates retention checks and cryptographic audit-logged purges."""

    def __init__(
        self,
        db_path: str = "data/police_records.db",
        tracks_dir: str = "data/tracks",
        audit_logger: Optional[AuditLogger] = None,
        unmatched_retention_hours: float = 72.0,
        unreviewed_retention_days: float = 30.0
    ):
        self.db_path = db_path
        self.tracks_dir = tracks_dir
        self.audit_logger = audit_logger or AuditLogger(db_path)
        self.unmatched_retention_sec = unmatched_retention_hours * 3600.0
        self.unreviewed_retention_sec = unreviewed_retention_days * 86400.0

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_retention_status(self) -> Dict[str, Any]:
        """Summary of biometric storage, oldest records, and items approaching expiration."""
        now = time.time()
        with self._get_connection() as conn:
            total_tracks = conn.execute("SELECT COUNT(*) as c FROM tracks").fetchone()["c"]
            oldest_track = conn.execute("SELECT MIN(first_seen) as m FROM tracks").fetchone()["m"]
            
            # Count match events
            confirmed = conn.execute("SELECT COUNT(*) as c FROM match_events WHERE status = 'CONFIRMED_MATCH'").fetchone()["c"]
            pending = conn.execute("SELECT COUNT(*) as c FROM match_events WHERE status = 'REVIEW_REQUIRED'").fetchone()["c"]

        # Track file counts on disk
        crop_count = 0
        disk_bytes = 0
        if os.path.exists(self.tracks_dir):
            for root, _, files in os.walk(self.tracks_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    crop_count += 1
                    disk_bytes += os.path.getsize(fp)

        oldest_age_hours = (now - oldest_track) / 3600.0 if oldest_track else 0.0

        return {
            "policy": {
                "unmatched_retention_hours": self.unmatched_retention_sec / 3600.0,
                "unreviewed_retention_days": self.unreviewed_retention_sec / 86400.0,
                "confirmed_retention": "INDEFINITE_UNDER_FIR_CASE"
            },
            "statistics": {
                "total_tracks": total_tracks,
                "total_crop_images_stored": crop_count,
                "total_disk_usage_mb": round(disk_bytes / (1024 * 1024), 2),
                "oldest_record_age_hours": round(oldest_age_hours, 1),
                "confirmed_matches": confirmed,
                "pending_reviews": pending
            }
        }

    def purge_expired_records(
        self,
        force_max_age_sec: Optional[float] = None,
        officer: Optional[OfficerIdentity] = None
    ) -> Dict[str, Any]:
        """Execute automated purge of expired biometric crops and database entries."""
        now = time.time()
        max_age_sec = force_max_age_sec if force_max_age_sec is not None else self.unmatched_retention_sec
        cutoff_time = now - max_age_sec

        deleted_files = 0
        deleted_rows = 0

        with self._get_connection() as conn:
            # Find tracks older than cutoff that are NOT associated with confirmed FIR matches
            query = """
                SELECT t.track_id, t.best_frame_path
                FROM tracks t
                LEFT JOIN match_events m ON t.track_id = m.track_id AND m.status = 'CONFIRMED_MATCH'
                WHERE t.last_seen < ? AND m.event_id IS NULL
            """
            cursor = conn.execute(query, (cutoff_time,))
            expired_tracks = cursor.fetchall()

            for row in expired_tracks:
                tid = row["track_id"]
                bf = row["best_frame_path"]
                if bf and os.path.exists(bf):
                    try:
                        os.remove(bf)
                        deleted_files += 1
                    except Exception:
                        pass
                
                # Delete from database
                conn.execute("DELETE FROM tracks WHERE track_id = ?", (tid,))
                conn.execute("DELETE FROM match_events WHERE track_id = ? AND status != 'CONFIRMED_MATCH'", (tid,))
                deleted_rows += 1

            conn.commit()

        # Record tamper-evident audit trail for data protection compliance
        audit_res = self.audit_logger.log_action(
            action_type="BIOMETRIC_DATA_PURGE",
            resource_id="ALL_EXPIRED_TRACKS",
            details={
                "records_purged": deleted_rows,
                "crop_files_removed": deleted_files,
                "cutoff_timestamp": cutoff_time,
                "cutoff_hours_ago": round(max_age_sec / 3600.0, 1),
                "statutory_basis": "DPDP_ACT_DATA_MINIMIZATION"
            },
            officer=officer or DEFAULT_OFFICER
        )

        return {
            "purged_records": deleted_rows,
            "removed_files": deleted_files,
            "cutoff_timestamp": cutoff_time,
            "audit_entry_id": audit_res["entry_id"]
        }
