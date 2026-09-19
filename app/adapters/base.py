"""Base interfaces and data structures for pluggable perception model adapters.

Supports seamless switching across open-source candidates:
- Detection: YOLOv8/11, RT-DETR
- Tracking: ByteTrack, BoT-SORT, NVIDIA DeepStream NvTracker, MMTracking
- Person Re-ID: Torchreid OSNet, FastReID (SBS/BoT/AGW)
- Face Analysis: InsightFace ArcFace with 3-Tier CCTV decomposition
- Pose Estimation: MMPose, RTMPose
- Gait Recognition: OpenGait (GaitBase/GaitGL/GaitSet), GaitSet dynamics
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


@dataclass
class ModelLicenseInfo:
    """License and governance metadata for open-source models and pretrained weights."""
    model_id: str
    component_name: str
    capability: str
    code_license: str  # e.g., "MIT", "Apache-2.0", "AGPL-3.0", "Academic Non-Commercial"
    weights_license: str  # e.g., "MIT", "Apache-2.0", "Non-Commercial Research Only"
    is_commercial_ready: bool
    restriction_notice: str
    official_repo: str


@dataclass
class DetectionResult:
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    class_name: str = "person"
    track_id: Optional[int] = None
    mask: Optional[np.ndarray] = None


@dataclass
class TrackingResult:
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    state: str = "CONFIRMED"  # TENTATIVE, CONFIRMED, LOST
    time_since_update: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)


@dataclass
class FaceAnalysisResult:
    bbox: Tuple[int, int, int, int]
    quality_score: float
    is_available: bool
    status: str  # FULL_FACE, PARTIAL_UPPER, MASKED_LOWER, BLURRED, UNAVAILABLE
    landmarks: Optional[List[Tuple[float, float]]] = None
    tier_embeddings: Dict[str, List[float]] = field(default_factory=dict)
    full_embedding: List[float] = field(default_factory=list)


@dataclass
class PoseResult:
    keypoints: List[Tuple[float, float, float]]  # (x, y, confidence) for 17 joints
    spine_tilt_deg: float = 0.0
    posture_score: float = 0.85
    bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class ReIDResult:
    embedding: List[float]
    model_name: str
    feature_dim: int = 512
    quality_score: float = 1.0


@dataclass
class GaitSequenceResult:
    gait_embedding: List[float]
    stride_length_cm: float
    cadence_steps_per_sec: float
    posture_score: float
    spine_tilt_deg: float
    quality_score: float = 1.0
    model_backend: str = "GAIT_ENGINE"
    raw_waveform: List[float] = field(default_factory=list)


class BaseDetector(ABC):
    @abstractmethod
    def detect_and_track(self, frame: np.ndarray, frame_id: int) -> List[DetectionResult]:
        """Detect and track persons in frame."""
        pass

    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        """Return backend metadata and configuration."""
        pass


class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections: List[DetectionResult], frame_id: int) -> List[TrackingResult]:
        """Update multi-target track associations across frames."""
        pass

    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        """Return tracker configuration and status."""
        pass


class BaseReIDModel(ABC):
    @abstractmethod
    def extract_embedding(self, person_crop: np.ndarray) -> ReIDResult:
        """Extract L2-normalized body Re-ID embedding."""
        pass

    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        pass


class BaseFaceAnalyzer(ABC):
    @abstractmethod
    def analyze_face(self, person_crop: np.ndarray) -> FaceAnalysisResult:
        """Extract quality-aware 3-tier face decomposition and embeddings."""
        pass

    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        pass


class BasePoseEstimator(ABC):
    @abstractmethod
    def estimate_pose(self, person_crop: np.ndarray) -> PoseResult:
        """Extract skeletal joints and posture dynamics."""
        pass

    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        pass


class BaseGaitModel(ABC):
    @abstractmethod
    def extract_sequence(self, track_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and extract valid temporal sequence for gait analysis."""
        pass

    @abstractmethod
    def generate_embedding(self, sequence: List[Dict[str, Any]], estimated_height_cm: float = 170.0) -> List[float]:
        """Generate normalized gait embedding representation."""
        pass

    @abstractmethod
    def quality_score(self, sequence: List[Dict[str, Any]]) -> float:
        """Compute confidence/quality score of the gait sequence."""
        pass

    @abstractmethod
    def analyze_sequence(self, track_history: List[Dict[str, Any]], estimated_height_cm: float = 170.0) -> Dict[str, Any]:
        """Complete analysis returning gait parameters and embedding."""
        pass

    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        pass
