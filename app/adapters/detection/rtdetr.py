"""RT-DETR (Real-Time DEtection TRansformer) Detector Adapter.

RT-DETR replaces anchor-based NMS with an end-to-end transformer decoder,
offering high accuracy without post-processing bottlenecks.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from app.adapters.base import DetectionResult
from app.adapters.detection.base import BasePedestrianDetector
from app.detection.person_detector import PersonDetector


class RTDETRDetectorAdapter(BasePedestrianDetector):
    """Adapter for Baidu / Ultralytics RT-DETR transformer detector."""

    def __init__(
        self,
        model_name: str = "rtdetr-l.pt",
        conf_thresh: float = 0.35,
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.conf_thresh = conf_thresh
        self.device = device
        self.rtdetr_model = None
        self.backend = "FALLBACK_OPENCV"
        self.fallback_detector = PersonDetector(confidence_threshold=conf_thresh)

        try:
            from ultralytics import RTDETR  # type: ignore
            self.rtdetr_model = RTDETR(model_name)
            self.backend = f"RT-DETR_{model_name.upper().replace('.PT', '')}"
        except Exception:
            self.backend = f"OPENCV_HOG_PEDESTRIAN_DETECTOR (RT-DETR-Fallback: {model_name})"

    def detect_and_track(self, frame: np.ndarray, frame_id: int) -> List[DetectionResult]:
        """Detect persons using RT-DETR transformer set prediction."""
        results: List[DetectionResult] = []

        if self.rtdetr_model is not None:
            try:
                preds = self.rtdetr_model.predict(
                    frame,
                    classes=[0],
                    conf=self.conf_thresh,
                    device=self.device,
                    verbose=False
                )
                if preds and len(preds) > 0:
                    boxes = preds[0].boxes
                    if boxes is not None:
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
        return {
            "model_id": "rtdetr",
            "model_name": self.model_name,
            "backend": self.backend,
            "architecture": "Vision Transformer + Hybrid Encoder + NMS-Free Decoder",
            "conf_thresh": self.conf_thresh,
            "is_neural_model_loaded": self.rtdetr_model is not None
        }
