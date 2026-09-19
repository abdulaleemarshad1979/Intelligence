"""Person detection module optimized for CCTV surveillance feeds."""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

class PersonDetector:
    def __init__(self, confidence_threshold: float = 0.35, min_height: int = 70, min_width: int = 25):
        self.confidence_threshold = confidence_threshold
        self.min_height = min_height
        self.min_width = min_width
        
        # Initialize OpenCV Default HOG Pedestrian Detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Initialize Background Subtractor for CCTV motion segmentation
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=True)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect persons in frame. Returns list of bounding boxes with confidence.
        
        Format: [{'box': [x1, y1, x2, y2], 'confidence': float, 'source': str}]
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        detections: List[Dict[str, Any]] = []

        # 1. HOG Pedestrian Detector
        # Resize frame slightly for speed and better scale detection
        scale_w = min(1.0, 800.0 / w)
        if scale_w < 1.0:
            small_frame = cv2.resize(frame, (int(w * scale_w), int(h * scale_w)))
            scale = 1.0 / scale_w
        else:
            small_frame = frame
            scale = 1.0

        boxes, weights = self.hog.detectMultiScale(
            small_frame,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )

        for (bx, by, bw, bh), weight in zip(boxes, weights):
            conf = float(weight[0]) if hasattr(weight, '__len__') else float(weight)
            # Map back to original coordinate system
            x1 = int(bx * scale)
            y1 = int(by * scale)
            x2 = int((bx + bw) * scale)
            y2 = int((by + bh) * scale)

            # Clamp coordinates
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            box_h = y2 - y1
            box_w = x2 - x1

            if box_h >= self.min_height and box_w >= self.min_width and (box_h / max(1, box_w)) > 1.2:
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": min(1.0, max(0.4, 0.5 + conf * 0.2)),
                    "source": "HOG"
                })

        # 2. CCTV Motion Segmentation (MOG2) to catch pedestrians missed by HOG
        fg_mask = self.bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1200:
                continue
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = ch / max(1, cw)
            
            # Pedestrians in CCTV have vertical aspect ratio (height > 1.4 * width)
            if ch >= self.min_height and cw >= self.min_width and 1.3 <= aspect_ratio <= 4.2:
                # Check overlap with existing HOG detections
                overlaps = False
                for d in detections:
                    ex1, ey1, ex2, ey2 = d["box"]
                    # Calculate IoU
                    ix1 = max(cx, ex1)
                    iy1 = max(cy, ey1)
                    ix2 = min(cx + cw, ex2)
                    iy2 = min(cy + ch, ey2)
                    iw = max(0, ix2 - ix1)
                    ih = max(0, iy2 - iy1)
                    intersection = iw * ih
                    union = (cw * ch) + ((ex2 - ex1) * (ey2 - ey1)) - intersection
                    if union > 0 and (intersection / union) > 0.2:
                        overlaps = True
                        break

                if not overlaps:
                    detections.append({
                        "box": [cx, cy, min(w - 1, cx + cw), min(h - 1, cy + ch)],
                        "confidence": 0.65,
                        "source": "MOTION_SEG"
                    })

        # Apply Non-Maximum Suppression (NMS)
        return self._apply_nms(detections, iou_thresh=0.35)

    def _apply_nms(self, detections: List[Dict[str, Any]], iou_thresh: float) -> List[Dict[str, Any]]:
        if not detections:
            return []

        boxes = np.array([d["box"] for d in detections], dtype=float)
        scores = np.array([d["confidence"] for d in detections], dtype=float)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]

        return [detections[k] for k in keep]
