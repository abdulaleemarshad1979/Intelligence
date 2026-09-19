"""Vision package: RT-DETR detection, ByteTrack tracking, UniPAR attribute parsing, OSNet Re-ID, and YuNet/SFace biometrics."""
from app.vision.detector import RTDETRDetector, Detection
from app.vision.tracker import ByteTrackTracker, Tracklet
from app.vision.par_engine import AttributeParsingEngine
from app.vision.reid_engine import OSNetReIDEngine
from app.vision.face_engine import FaceBiometricEngine

__all__ = [
    "RTDETRDetector",
    "Detection",
    "ByteTrackTracker",
    "Tracklet",
    "AttributeParsingEngine",
    "OSNetReIDEngine",
    "FaceBiometricEngine",
]
