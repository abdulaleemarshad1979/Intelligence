"""Detection adapters package."""

from app.adapters.detection.base import BasePedestrianDetector
from app.adapters.detection.yolo import YOLODetectorAdapter
from app.adapters.detection.rtdetr import RTDETRDetectorAdapter

__all__ = ["BasePedestrianDetector", "YOLODetectorAdapter", "RTDETRDetectorAdapter"]
