"""Detection adapters package supporting SOTA full-body pedestrian models."""

from app.adapters.detection.base import BasePedestrianDetector
from app.adapters.detection.yolo import YOLODetectorAdapter
from app.adapters.detection.rtdetr import RTDETRDetectorAdapter
from app.adapters.detection.fullbody_pose import FullBodyPoseDetectorAdapter
from app.adapters.detection.ensemble import EnsemblePedestrianDetector

__all__ = [
    "BasePedestrianDetector",
    "YOLODetectorAdapter",
    "RTDETRDetectorAdapter",
    "FullBodyPoseDetectorAdapter",
    "EnsemblePedestrianDetector"
]
