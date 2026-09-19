"""Investigation Ontology: Person / Target of Interest.

Represents a real-world canonical subject or target of interest in a Gotham-style
investigation graph. Distinct from transient camera tracks or single observations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time
import uuid

@dataclass
class TargetProfile:
    """Canonical Person / Target entity."""
    person_id: str = field(default_factory=lambda: f"POI-{uuid.uuid4().hex[:8].upper()}")
    target_code: str = "TARGET-001"
    canonical_name: str = "UNIDENTIFIED_TARGET"
    status: str = "PERSON_OF_INTEREST"  # UNIDENTIFIED_TARGET, PERSON_OF_INTEREST, CONFIRMED_SUSPECT, CLEARED
    notes: str = ""
    linked_tracks: List[str] = field(default_factory=list)
    confirmed_identity: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def link_track(self, track_id: str):
        if track_id not in self.linked_tracks:
            self.linked_tracks.append(track_id)
            self.updated_at = time.time()

    def confirm_identity(self, name: str, fir_no: str, officer_badge: str, notes: str = ""):
        self.canonical_name = name
        self.status = "CONFIRMED_SUSPECT"
        self.confirmed_identity = {
            "name": name,
            "fir_no": fir_no,
            "adjudicated_by": officer_badge,
            "adjudicated_at": time.time(),
            "notes": notes
        }
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "person_id": self.person_id,
            "target_code": self.target_code,
            "canonical_name": self.canonical_name,
            "status": self.status,
            "notes": self.notes,
            "linked_tracks": self.linked_tracks,
            "confirmed_identity": self.confirmed_identity,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
