"""Human-in-the-Loop Review Gate for Forensic Candidate Sign-Off.

Mandatory gate ensuring no automated surveillance match is escalated without explicit,
authenticated human investigator verification with badge number and case notes.
"""

import sqlite3
import time
from typing import Dict, Any, Optional
from pydantic import BaseModel
from app.compliance.rbac import OfficerIdentity, Permission, DEFAULT_OFFICER
from app.compliance.audit import AuditLogger

class ReviewSubmission(BaseModel):
    event_id: str
    verdict: str  # CONFIRMED_MATCH, REJECTED_FALSE_POSITIVE, INCONCLUSIVE
    investigator_notes: str
    badge_number: Optional[str] = None
    officer_name: Optional[str] = None
    fir_case_number: Optional[str] = None

class HumanReviewGate:
    """Manages candidate verification workflows and officer sign-offs."""

    def __init__(self, db_path: str = "data/police_records.db", audit_logger: Optional[AuditLogger] = None):
        self.db_path = db_path
        self.audit_logger = audit_logger or AuditLogger(db_path)
        self._ensure_review_columns()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_review_columns(self):
        """Add verification metadata columns to match_events if missing."""
        with self._get_connection() as conn:
            cursor = conn.execute("PRAGMA table_info(match_events)")
            cols = [r["name"] for r in cursor.fetchall()]
            if "reviewed_by_officer" not in cols:
                conn.execute("ALTER TABLE match_events ADD COLUMN reviewed_by_officer TEXT DEFAULT ''")
            if "reviewer_badge" not in cols:
                conn.execute("ALTER TABLE match_events ADD COLUMN reviewer_badge TEXT DEFAULT ''")
            if "review_notes" not in cols:
                conn.execute("ALTER TABLE match_events ADD COLUMN review_notes TEXT DEFAULT ''")
            if "reviewed_at" not in cols:
                conn.execute("ALTER TABLE match_events ADD COLUMN reviewed_at REAL DEFAULT 0.0")
            conn.commit()

    def submit_review(
        self,
        submission: ReviewSubmission,
        officer: Optional[OfficerIdentity] = None
    ) -> Dict[str, Any]:
        """Process official officer sign-off on a candidate match."""
        active_officer = officer or DEFAULT_OFFICER
        if not active_officer.can(Permission.SIGN_OFF_MATCH):
            raise PermissionError(f"Role {active_officer.role} does not have authority to sign off on candidate matches.")

        badge = submission.badge_number or active_officer.badge_number
        name = submission.officer_name or active_officer.full_name
        now = time.time()

        with self._get_connection() as conn:
            # Check event exists
            cur = conn.execute("SELECT * FROM match_events WHERE event_id = ?", (submission.event_id,))
            event = cur.fetchone()
            if not event:
                raise ValueError(f"Match event {submission.event_id} not found.")

            # Update match status & verification notes
            conn.execute("""
                UPDATE match_events
                SET status = ?,
                    reviewed_by_officer = ?,
                    reviewer_badge = ?,
                    review_notes = ?,
                    reviewed_at = ?
                WHERE event_id = ?
            """, (submission.verdict, name, badge, submission.investigator_notes, now, submission.event_id))
            conn.commit()

        # Log cryptographic audit entry
        audit_res = self.audit_logger.log_action(
            action_type="HUMAN_REVIEW_SIGNOFF",
            resource_id=submission.event_id,
            details={
                "suspect_id": event["suspect_id"],
                "suspect_name": event["suspect_name"],
                "track_id": event["track_id"],
                "previous_status": event["status"],
                "new_verdict": submission.verdict,
                "notes": submission.investigator_notes,
                "fir_no": event["fir_no"]
            },
            officer=active_officer
        )

        return {
            "success": True,
            "event_id": submission.event_id,
            "verdict": submission.verdict,
            "reviewed_by": f"{name} ({badge})",
            "reviewed_at": now,
            "audit_entry_id": audit_res["entry_id"]
        }
