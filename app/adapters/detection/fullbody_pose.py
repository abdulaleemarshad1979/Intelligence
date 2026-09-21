"""SOTA Full-Body Pose & Keypoint Pedestrian Detector Adapter.

Simultaneously detects the full-body pedestrian bounding box AND extracts 17 COCO
anatomical skeletal landmarks:
- Head: Nose, Eyes, Ears
- Upper Body: Shoulders, Elbows, Wrists
- Lower Body: Hips, Knees, Ankles

Supports YOLO11-Pose (yolo11x-pose, yolo11l-pose, yolo11n-pose) and YOLOv8-Pose (yolov8x-pose, yolov8n-pose).
"""

import os
import cv2
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from app.adapters.base import DetectionResult
from app.adapters.detection.base import BasePedestrianDetector
from app.detection.person_detector import PersonDetector

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FullBodyPoseDetectorAdapter(BasePedestrianDetector):
    """Unified detector extracting both full-body bounding boxes and 17 skeletal keypoints."""

    def __init__(
        self,
        model_name: str = "yolov8n-pose.pt",
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device or get_default_device()
        self.pose_model = None
        self.backend = "FALLBACK_OPENCV"
        self.fallback_detector = PersonDetector(confidence_threshold=conf_thresh, enable_pose=True)

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        local_p = os.path.join(base_dir, "models", model_name)
        weights_to_load = local_p if os.path.isfile(local_p) else (model_name if os.path.isfile(model_name) else None)

        if weights_to_load and os.path.isfile(weights_to_load):
            try:
                from ultralytics import YOLO  # type: ignore
                self.pose_model = YOLO(weights_to_load)
                model_tag = model_name.upper().replace(".PT", "")
                self.backend = f"ULTRALYTICS_POSE_{model_tag}"
            except Exception:
                self.backend = f"FALLBACK_POSE_DETECTOR ({model_name})"
        else:
            self.backend = f"FALLBACK_POSE_DETECTOR ({model_name})"

    def detect_and_track(self, frame: np.ndarray, frame_id: int) -> List[DetectionResult]:
        """Detect full-body pedestrians with 17 skeletal landmarks."""
        results: List[DetectionResult] = []

        if self.pose_model is not None:
            try:
                preds = self.pose_model.predict(
                    frame,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    device=self.device,
                    verbose=False
                )
                if preds and len(preds) > 0:
                    pred_item = preds[0]
                    boxes = pred_item.boxes
                    keypoints_data = None
                    if hasattr(pred_item, "keypoints") and pred_item.keypoints is not None:
                        try:
                            keypoints_data = pred_item.keypoints.data.cpu().numpy()
                        except Exception:
                            keypoints_data = None

                    if boxes is not None and len(boxes) > 0:
                        for idx, box in enumerate(boxes):
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.8
                            w = max(1, x2 - x1)
                            h = max(1, y2 - y1)

                            # Format 17 anatomical keypoints
                            kpts_dict: Dict[str, Any] = {}
                            if keypoints_data is not None and idx < len(keypoints_data):
                                raw_pts = keypoints_data[idx]
                                for k_i, k_name in enumerate(COCO_KEYPOINT_NAMES):
                                    if k_i < len(raw_pts):
                                        pt = raw_pts[k_i]
                                        px, py = float(pt[0]), float(pt[1])
                                        p_conf = float(pt[2]) if len(pt) > 2 else 1.0
                                        kpts_dict[k_name] = {
                                            "global": [round(px, 1), round(py, 1)],
                                            "crop": [round(max(0.0, px - x1), 1), round(max(0.0, py - y1), 1)],
                                            "confidence": round(p_conf, 3)
                                        }

                            results.append(DetectionResult(
                                bbox=(x1, y1, w, h),
                                confidence=round(conf, 3),
                                class_name="person",
                                track_id=None,
                                keypoints=kpts_dict if kpts_dict else None
                            ))
                        return results
            except Exception:
                pass

        # Fallback detection
        dets = self.fallback_detector.detect(frame)
        for d in dets:
            box = d.get("box", [0, 0, 0, 0]) if isinstance(d, dict) else getattr(d, "box", [0, 0, 0, 0])
            conf = d.get("confidence", 0.5) if isinstance(d, dict) else getattr(d, "confidence", 0.5)
            kpts = d.get("keypoints", None) if isinstance(d, dict) else getattr(d, "keypoints", None)

            x1, y1, x2, y2 = box
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            results.append(DetectionResult(
                bbox=(x1, y1, w, h),
                confidence=round(conf, 3),
                class_name="person",
                track_id=1,
                keypoints=kpts
            ))
        return results

    def get_backend_info(self) -> Dict[str, Any]:
        return {
            "model_id": "fullbody_pose",
            "model_name": self.model_name,
            "backend": self.backend,
            "device": self.device,
            "num_keypoints": 17,
            "architecture": "Full-Body 17-Keypoint Skeletal + Bounding Box Network",
            "conf_thresh": self.conf_thresh,
            "iou_thresh": self.iou_thresh,
            "is_neural_model_loaded": self.pose_model is not None
        }
