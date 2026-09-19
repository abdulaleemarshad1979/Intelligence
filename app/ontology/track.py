"""Investigation Ontology: Track Entity.

Represents a continuous spatio-temporal detection sequence on a specific CCTV camera.
Tracks remain separate observation records that can be linked to other tracks or targets.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time

@dataclass
class TrackEntity:
    """Camera-specific continuous observation track."""
    track_id: str
    camera_id: str
    start_time: float
    end_time: float
    frame_count: int = 1
    best_frame_path: str = ""
    direction: str = "NORTH"  # NORTH, SOUTH, EAST, WEST, STATIONARY, etc.
    movement_vector: List[float] = field(default_factory=lambda: [0.0, 1.0])  # dx, dy
    status: str = "ACTIVE"  # ACTIVE, ARCHIVED, ASSOCIATED
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_sec": self.duration_sec,
            "frame_count": self.frame_count,
            "best_frame_path": self.best_frame_path,
            "direction": self.direction,
            "movement_vector": self.movement_vector,
            "status": self.status,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
