"""Data models and structures for Multi-Camera Target Person Search."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
import time


class TargetSearchMode(str, Enum):
    AUTO = "auto"
    FACE = "face"
    REID = "reid"


class CandidateStatus(str, Enum):
    OBSERVED = "OBSERVED"
    TRACKING = "TRACKING"
    CONFIRMED_CANDIDATE = "CONFIRMED_CANDIDATE"
    EXPIRED = "EXPIRED"
    REVIEWED = "REVIEWED"


class SessionStatus(str, Enum):
    INITIALIZING = "Initializing"
    RUNNING = "Running"
    STOPPED = "Stopped"
    COMPLETED = "Completed"
    ERROR = "Error"


@dataclass
class TargetSearchConfig:
    min_confirmations: int = 4
    confirmation_window_sec: float = 8.0
    minimum_raw_similarity: float = 0.55
    minimum_quality_score: float = 0.35
    duplicate_event_cooldown_sec: float = 15.0


@dataclass
class Observation:
    timestamp: float
    camera_id: str
    track_id: str
    bbox: List[int]  # [x, y, w, h]
    raw_similarity: float
    quality_score: float
    detection_score: float
    frame_id: int
    face_bbox: Optional[List[int]] = None
    similarity_type: str = "face"  # "face" or "reid"
    best_frame_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "bbox": self.bbox,
            "raw_similarity": round(float(self.raw_similarity), 4),
            "quality_score": round(float(self.quality_score), 4),
            "detection_score": round(float(self.detection_score), 4),
            "frame_id": self.frame_id,
            "face_bbox": self.face_bbox,
            "similarity_type": self.similarity_type,
            "best_frame_score": round(float(self.best_frame_score), 4)
        }


@dataclass
class CandidateEvent:
    event_id: str
    session_id: str
    camera_id: str
    track_id: str
    first_seen: float
    last_seen: float
    confirmation_count: int
    raw_similarity_max: float
    raw_similarity_mean: float
    quality_mean: float
    confirmation_score: float
    status: str = CandidateStatus.CONFIRMED_CANDIDATE.value
    review_required: bool = True
    best_frame_path: str = ""
    person_crop_path: str = ""
    face_crop_path: Optional[str] = None
    hashes: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "confirmation_count": self.confirmation_count,
            "raw_similarity_max": round(float(self.raw_similarity_max), 4),
            "raw_similarity_mean": round(float(self.raw_similarity_mean), 4),
            "quality_mean": round(float(self.quality_mean), 4),
            "confirmation_score": round(float(self.confirmation_score), 4),
            "status": self.status,
            "review_required": self.review_required,
            "best_frame_path": self.best_frame_path,
            "person_crop_path": self.person_crop_path,
            "face_crop_path": self.face_crop_path,
            "hashes": self.hashes,
            "metadata": self.metadata,
            "created_at": self.created_at
        }


@dataclass
class TargetSearchSession:
    session_id: str
    target_id: str
    name: str
    requested_mode: str
    active_mode: str
    selected_cameras: List[str]
    reference_image_path: str
    reference_face_embedding: Optional[List[float]] = None
    reference_reid_embedding: Optional[List[float]] = None
    reference_face_bbox: Optional[List[int]] = None
    reference_quality: Dict[str, Any] = field(default_factory=dict)
    config: TargetSearchConfig = field(default_factory=TargetSearchConfig)
    status: str = SessionStatus.RUNNING.value
    created_at: float = field(default_factory=time.time)
    stopped_at: Optional[float] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "target_id": self.target_id,
            "name": self.name,
            "requested_mode": self.requested_mode,
            "active_mode": self.active_mode,
            "selected_cameras": self.selected_cameras,
            "reference_image_path": self.reference_image_path,
            "reference_quality": self.reference_quality,
            "status": self.status,
            "config": {
                "min_confirmations": self.config.min_confirmations,
                "confirmation_window_sec": self.config.confirmation_window_sec,
                "minimum_raw_similarity": self.config.minimum_raw_similarity,
                "minimum_quality_score": self.config.minimum_quality_score,
                "duplicate_event_cooldown_sec": self.config.duplicate_event_cooldown_sec
            },
            "created_at": self.created_at,
            "stopped_at": self.stopped_at,
            "error_message": self.error_message
        }
