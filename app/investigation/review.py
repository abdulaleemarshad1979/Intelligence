"""Human Review & Consequential Adjudication Workflow.

Enforces human-in-the-loop review for high-consequence investigative decisions:
Confirming candidate associations, establishing legal identities, or rejecting false links.
Integrates with tamper-evident audit logging.
"""

from typing import Dict, Any, Optional
import time
import uuid

from app.database.repository import Repository
from app.database.models import AdjudicationReview, RelationshipLink

class HumanAdjudicationGate:
    """Enforces officer sign-off and legal defensibility on candidate links."""

    def __init__(self, repo: Repository):
        self.repo = repo

    def submit_review(
        self,
        relationship_id: str,
        reviewer_badge: str,
        reviewer_name: str,
        decision: str,  # CONFIRM_IDENTITY, REJECT_ASSOCIATION, DEFER_REVIEW
        review_notes: str = "",
        target_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Record human adjudication decision and update link status."""
        rev_id = f"REV-{uuid.uuid4().hex[:8].upper()}"
        review = AdjudicationReview(
            review_id=rev_id,
            relationship_id=relationship_id,
            target_id=target_id,
            reviewer_badge=reviewer_badge,
            reviewer_name=reviewer_name,
            decision=decision,
            review_notes=review_notes,
            timestamp=time.time()
        )
        self.repo.save_review(review)

        # Update relationship status
        if decision == "CONFIRM_IDENTITY":
            new_status = "CONFIRMED"
        elif decision == "REJECT_ASSOCIATION":
            new_status = "REJECTED"
        else:
            new_status = "REVIEW_PENDING"

        self.repo.update_relationship_status(relationship_id, new_status)

        # Log in tamper-evident audit log
        self.repo.log_audit(
            action="HUMAN_ADJUDICATION_SUBMITTED",
            details=f"Officer {reviewer_name} ({reviewer_badge}) adjudicated {relationship_id} -> {decision}: {review_notes}",
            operator=reviewer_badge
        )

        return {
            "review_id": rev_id,
            "relationship_id": relationship_id,
            "decision": decision,
            "status": new_status,
            "adjudicated_by": f"{reviewer_name} ({reviewer_badge})",
            "timestamp": review.timestamp
        }
