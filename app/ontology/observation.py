"""Investigation Ontology: Observation Entity.

Represents a single discrete frame capture or sensor observation associated with
a camera track, providing visual crops, bounding coordinates, and timestamps.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import time
import uuid

@dataclass
class ObservationEntity:
    """Discrete CCTV visual observation."""
    observation_id: str = field(default_factory=lambda: f"OBS-{uuid.uuid4().hex[:8].upper()}")
    track_id: str = ""
    camera_id: str = ""
    timestamp: float = field(default_factory=time.time)
    frame_number: int = 0
    crop_path: str = ""
    bbox: Dict[str, Any] = field(default_factory=lambda: {"x": 0, "y": 0, "w": 100, "h": 200})
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "crop_path": self.crop_path,
            "bbox": self.bbox,
            "created_at": self.created_at
        }
