"""Detector & Tracker Adapter supporting YOLOv8 / YOLO11 + ByteTrack / BoT-SORT.

Provides unified detection and tracking with automated fallback to the OpenCV
pedestrian engine if Ultralytics weights or CUDA are not initialized.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from app.adapters.base import BaseDetector, DetectionResult
from app.detection.person_detector import PersonDetector
from app.tracking.tracker import MultiPersonTracker

class YOLOByteTrackAdapter(BaseDetector):
    """Adapter for YOLO + ByteTrack/BoT-SORT detection & tracking pipeline."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        tracker_type: str = "bytetrack",  # bytetrack or botsort
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        use_fallback_if_unavailable: bool = True
    ):
        self.model_name = model_name
        self.tracker_type = tracker_type
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.backend = "FALLBACK_OPENCV"
        self.yolo_model = None
        
        # Check if ultralytics is available
        try:
            from ultralytics import YOLO  # type: ignore
            self.yolo_model = YOLO(model_name)
            self.backend = f"YOLO_{model_name}_{tracker_type.upper()}"
        except Exception:
            # Graceful fallback to lightweight pedestrian detector + multi-tracker
            if not use_fallback_if_unavailable:
                raise
            self.backend = "LIGHTWEIGHT_OPENCV_CENTROID"
            self.fallback_detector = PersonDetector(confidence_threshold=conf_thresh)
            self.fallback_tracker = MultiPersonTracker(max_age=30, min_hits=2, iou_threshold=iou_thresh)

    def detect_and_track(self, frame: np.ndarray, frame_id: int) -> List[DetectionResult]:
        """Detect and track persons, assigning persistent track IDs across frames."""
        results: List[DetectionResult] = []

        if self.yolo_model is not None:
            try:
                # Run YOLO tracking with ByteTrack / BoT-SORT
                yolo_results = self.yolo_model.track(
                    frame,
                    persist=True,
                    classes=[0],  # Class 0 is person in COCO
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    tracker=f"{self.tracker_type}.yaml",
                    verbose=False
                )
                if yolo_results and len(yolo_results) > 0:
                    boxes = yolo_results[0].boxes
                    if boxes is not None:
                        for box in boxes:
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            w = max(1, x2 - x1)
                            h = max(1, y2 - y1)
                            conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.8
                            track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None
                            results.append(DetectionResult(
                                bbox=(x1, y1, w, h),
                                confidence=conf,
                                class_name="person",
                                track_id=track_id
                            ))
                return results
            except Exception:
                pass  # Fall back if runtime inference error occurs

        # Fallback implementation
        detections = self.fallback_detector.detect(frame)
        active_tracks = self.fallback_tracker.update(detections)
        for trk in active_tracks:
            x1, y1, x2, y2 = trk.box
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            try:
                tid = int(str(trk.track_id).replace("TRACK-", ""))
            except Exception:
                tid = 1
            results.append(DetectionResult(
                bbox=(x1, y1, w, h),
                confidence=trk.confidence,
                class_name="person",
                track_id=tid
            ))
        return results

    def get_backend_info(self) -> Dict[str, Any]:
        return {
            "model_type": "YOLO / ByteTrack",
            "backend": self.backend,
            "tracker": self.tracker_type,
            "conf_thresh": self.conf_thresh,
            "iou_thresh": self.iou_thresh
        }
