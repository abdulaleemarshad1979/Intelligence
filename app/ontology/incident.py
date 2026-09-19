"""Investigation Ontology: Incident Entity.

Represents an investigative case or incident initiating target tracking workflows,
connecting incident time, origin camera, case facts, and seed observations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time
import uuid

@dataclass
class IncidentEntity:
    """Investigative Case / Incident."""
    incident_id: str = field(default_factory=lambda: f"INC-{uuid.uuid4().hex[:8].upper()}")
    case_number: str = "INC-2026-0041"
    title: str = "Suspect Track Initiation"
    description: str = "Person of interest observed leaving scene"
    camera_id: str = "CAM-017"
    incident_time: float = field(default_factory=time.time)
    status: str = "OPEN"  # OPEN, INVESTIGATING, ADJUDICATED, CLOSED
    priority: str = "HIGH"  # ROUTINE, MEDIUM, HIGH, CRITICAL
    officer_in_charge: str = "AP-EG-8821"
    seed_track_id: Optional[str] = None
    poi_target_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "case_number": self.case_number,
            "title": self.title,
            "description": self.description,
            "camera_id": self.camera_id,
            "incident_time": self.incident_time,
            "status": self.status,
            "priority": self.priority,
            "officer_in_charge": self.officer_in_charge,
            "seed_track_id": self.seed_track_id,
            "poi_target_id": self.poi_target_id,
            "created_at": self.created_at
        }
