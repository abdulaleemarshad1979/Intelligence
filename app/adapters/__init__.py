"""Model adapters package for pluggable perception architectures.

Exports all modular adapters:
- Detection: YOLODetectorAdapter, RTDETRDetectorAdapter, YOLOByteTrackAdapter
- Tracking: ByteTrackAdapter, BoTSORTAdapter, DeepStreamTrackerAdapter, MMTrackingAdapter
- Re-ID: OSNetReIDAdapter, FastReIDAdapter
- Face: InsightFaceArcFaceAdapter
- Pose: RTMPoseAdapter, MMPoseAdapter, MMPoseRTMPoseAdapter
- Gait: OpenGaitAdapter, GaitSetAdapter, GaitSetDynamicsAdapter
- Registry & Governance: model_registry, ModelRegistry, LicenseAuditor
"""

from app.adapters.base import (
    BaseDetector,
    BaseTracker,
    BaseReIDModel,
    BaseFaceAnalyzer,
    BasePoseEstimator,
    BaseGaitModel,
    DetectionResult,
    TrackingResult,
    FaceAnalysisResult,
    PoseResult,
    ReIDResult,
    GaitSequenceResult,
    ModelLicenseInfo
)

# Modular Subpackage Adapters
from app.adapters.detection import YOLODetectorAdapter, RTDETRDetectorAdapter
from app.adapters.tracking import (
    ByteTrackAdapter,
    BoTSORTAdapter,
    DeepStreamTrackerAdapter,
    MMTrackingAdapter
)
from app.adapters.reid import OSNetReIDAdapter, FastReIDAdapter
from app.adapters.face import InsightFaceArcFaceAdapter
from app.adapters.pose import RTMPoseAdapter, MMPoseAdapter
from app.adapters.gait import OpenGaitAdapter, GaitSetAdapter

# Registry & Governance
from app.adapters.license_audit import LicenseAuditor, MODEL_LICENSE_CATALOG
from app.adapters.registry import ModelRegistry, model_registry

# Backward-compatibility aliases for legacy imports
from app.adapters.detector_adapter import YOLOByteTrackAdapter
from app.adapters.pose_adapter import MMPoseRTMPoseAdapter
from app.adapters.gait_adapter import GaitSetDynamicsAdapter

__all__ = [
    # Base
    "BaseDetector",
    "BaseTracker",
    "BaseReIDModel",
    "BaseFaceAnalyzer",
    "BasePoseEstimator",
    "BaseGaitModel",
    "DetectionResult",
    "TrackingResult",
    "FaceAnalysisResult",
    "PoseResult",
    "ReIDResult",
    "GaitSequenceResult",
    "ModelLicenseInfo",
    # Detection
    "YOLODetectorAdapter",
    "RTDETRDetectorAdapter",
    "YOLOByteTrackAdapter",
    # Tracking
    "ByteTrackAdapter",
    "BoTSORTAdapter",
    "DeepStreamTrackerAdapter",
    "MMTrackingAdapter",
    # Re-ID
    "OSNetReIDAdapter",
    "FastReIDAdapter",
    # Face
    "InsightFaceArcFaceAdapter",
    # Pose
    "RTMPoseAdapter",
    "MMPoseAdapter",
    "MMPoseRTMPoseAdapter",
    # Gait
    "OpenGaitAdapter",
    "GaitSetAdapter",
    "GaitSetDynamicsAdapter",
    # Registry & Licensing
    "LicenseAuditor",
    "MODEL_LICENSE_CATALOG",
    "ModelRegistry",
    "model_registry"
]
