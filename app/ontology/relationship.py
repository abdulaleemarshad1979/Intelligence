"""Investigation Ontology: Relationship / Link Entity.

Defines directional and associative links between investigation objects
(e.g., Track-to-Track candidate associations, Track-to-Person linkages),
encapsulating full transparent evidence breakdowns and human adjudication states.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import time
import uuid

@dataclass
class RelationshipEntity:
    """Link between two ontology nodes."""
    relationship_id: str = field(default_factory=lambda: f"REL-{uuid.uuid4().hex[:8].upper()}")
    source_type: str = "TRACK"  # TRACK, PERSON, INCIDENT, OBSERVATION
    source_id: str = ""
    target_type: str = "TRACK"  # TRACK, PERSON, INCIDENT, OBSERVATION
    target_id: str = ""
    relationship_type: str = "CANDIDATE_SAME_PERSON"  # CANDIDATE_SAME_PERSON, CONFIRMED_SAME_PERSON, TRANSIT_STEP, ASSOCIATE
    confidence_score: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    status: str = "CANDIDATE"  # CANDIDATE, REVIEW_PENDING, CONFIRMED, REJECTED
    notes: str = ""
    adjudication_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def adjudicate(self, decision: str, reviewer_badge: str, notes: str = ""):
        self.status = "CONFIRMED" if decision == "CONFIRM_IDENTITY" else ("REJECTED" if decision == "REJECT_ASSOCIATION" else "REVIEW_PENDING")
        self.notes = notes
        self.adjudication_history.append({
            "decision": decision,
            "reviewer_badge": reviewer_badge,
            "notes": notes,
            "timestamp": time.time()
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relationship_id": self.relationship_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "confidence_score": self.confidence_score,
            "evidence": self.evidence,
            "status": self.status,
            "notes": self.notes,
            "adjudication_history": self.adjudication_history,
            "created_at": self.created_at
        }
