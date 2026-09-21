"""YOLO Pedestrian Detector Adapter supporting YOLO11 (SOTA 2024) and YOLOv8.

Supports the full scale spectrum:
- Heavyweight / Ultra: YOLO11x, YOLO11l, YOLOv8x (best-in-class mAP on pedestrians)
- Balanced: YOLO11m, YOLOv8m
- Real-time Edge: YOLO11s, YOLO11n, YOLOv8n

Uses Ultralytics inference with graceful fallback to the OpenCV pedestrian engine
if weights are not downloaded or running on headless CPU.
"""

import os
import cv2
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from app.adapters.base import DetectionResult
from app.adapters.detection.base import BasePedestrianDetector
from app.detection.person_detector import PersonDetector


def get_default_device() -> str:
    """Determine the optimal hardware accelerator available."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YOLODetectorAdapter(BasePedestrianDetector):
    """Adapter for YOLO11 / YOLOv8 SOTA full body pedestrian detection."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device or get_default_device()
        self.yolo_model = None
        self.backend = "FALLBACK_OPENCV"
        self.fallback_detector = PersonDetector(confidence_threshold=conf_thresh)

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        local_p = os.path.join(base_dir, "models", model_name)
        weights_to_load = local_p if os.path.isfile(local_p) else (model_name if os.path.isfile(model_name) else None)

        if weights_to_load and os.path.isfile(weights_to_load):
            try:
                from ultralytics import YOLO  # type: ignore
                self.yolo_model = YOLO(weights_to_load)
                model_tag = model_name.upper().replace(".PT", "")
                self.backend = f"ULTRALYTICS_{model_tag}"
            except Exception:
                self.backend = f"OPENCV_HOG_PEDESTRIAN_DETECTOR (YOLO-Fallback: {model_name})"
        else:
            self.backend = f"OPENCV_HOG_PEDESTRIAN_DETECTOR (YOLO-Fallback: {model_name})"

    def detect_and_track(self, frame: np.ndarray, frame_id: int) -> List[DetectionResult]:
        """Detect persons in frame."""
        results: List[DetectionResult] = []

        if self.yolo_model is not None:
            try:
                preds = self.yolo_model.predict(
                    frame,
                    classes=[0],  # COCO class 0 = person
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    device=self.device,
                    verbose=False
                )
                if preds and len(preds) > 0:
                    boxes = preds[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.8
                            w = max(1, x2 - x1)
                            h = max(1, y2 - y1)
                            results.append(DetectionResult(
                                bbox=(x1, y1, w, h),
                                confidence=round(conf, 3),
                                class_name="person",
                                track_id=None
                            ))
                        return results
            except Exception:
                pass

        # Fallback detection
        dets = self.fallback_detector.detect(frame)
        for d in dets:
            if isinstance(d, dict):
                box = d.get("box", [0, 0, 0, 0])
                conf = d.get("confidence", 0.5)
            else:
                box = getattr(d, "box", [0, 0, 0, 0])
                conf = getattr(d, "confidence", 0.5)

            x1, y1, x2, y2 = box
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            results.append(DetectionResult(
                bbox=(x1, y1, w, h),
                confidence=round(conf, 3),
                class_name="person",
                track_id=1
            ))
        return results

    def get_backend_info(self) -> Dict[str, Any]:
        arch = "C3k2 + SPPF (YOLO11)" if "11" in self.model_name else "CSPDarknet + PANet (YOLOv8)"
        return {
            "model_id": "yolo",
            "model_name": self.model_name,
            "backend": self.backend,
            "architecture": arch,
            "device": self.device,
            "conf_thresh": self.conf_thresh,
            "iou_thresh": self.iou_thresh,
            "is_neural_model_loaded": self.yolo_model is not None
        }
