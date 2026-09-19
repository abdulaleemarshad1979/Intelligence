"""Face detection and 3-Tier Partition (Upper, Mid, Lower) module for partial-face CCTV intelligence."""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

class FaceAnalyzer:
    def __init__(self):
        # Load Haar cascades as reliable, ultra-fast CPU fallback for face and eye detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def analyze_person_crop(self, person_crop: np.ndarray) -> Dict[str, Any]:
        """Analyzes a person crop to locate head/face and decompose into 3 tiers."""
        if person_crop is None or person_crop.size == 0:
            return self._empty_response("NO_IMAGE")

        ph, pw = person_crop.shape[:2]
        if ph < 50 or pw < 20:
            return self._empty_response("TOO_SMALL")

        # Head region is typically top 28% of person bounding box
        head_h = int(ph * 0.28)
        head_w = int(pw * 0.8)
        hx1 = max(0, int(pw * 0.1))
        hx2 = min(pw, hx1 + head_w)
        hy1 = 0
        hy2 = min(ph, head_h)

        head_crop = person_crop[hy1:hy2, hx1:hx2]
        gray_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)

        # 1. Check for frontal/profile face detection
        faces = self.face_cascade.detectMultiScale(gray_head, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        if len(faces) == 0:
            faces = self.profile_cascade.detectMultiScale(gray_head, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

        if len(faces) > 0:
            fx, fy, fw, fh = faces[0]
            # Refine face box in person crop coordinate space
            face_box = [hx1 + fx, hy1 + fy, hx1 + fx + fw, hy1 + fy + fh]
            face_crop = head_crop[fy:fy + fh, fx:fx + fw]
            detection_type = "CASCADE_MATCH"
        else:
            # Fallback: estimate face from head bounding box
            # Use color/texture contrast in head region
            fx, fy = int(pw * 0.18), int(ph * 0.02)
            fw, fh = int(pw * 0.64), int(ph * 0.24)
            face_box = [fx, fy, fx + fw, fy + fh]
            face_crop = person_crop[fy:fy + fh, fx:fx + fw]
            detection_type = "HEAD_GEOMETRY_ESTIMATE"

        if face_crop.size == 0 or face_crop.shape[0] < 12 or face_crop.shape[1] < 12:
            return self._empty_response("INVALID_FACE_CROP")

        # Decompose face into 3 Tiers:
        # Tier 1: UPPER (0% to 33%) -> Forehead / Eyes
        # Tier 2: MID   (33% to 66%) -> Nose / Cheekbones
        # Tier 3: LOWER (66% to 100%) -> Mouth / Jaw / Mask area
        fh_actual, fw_actual = face_crop.shape[:2]
        t1_end = int(fh_actual * 0.33)
        t2_end = int(fh_actual * 0.66)

        upper_crop = face_crop[0:t1_end, :]
        mid_crop = face_crop[t1_end:t2_end, :]
        lower_crop = face_crop[t2_end:fh_actual, :]

        upper_stat = self._evaluate_tier(upper_crop, "upper")
        mid_stat = self._evaluate_tier(mid_crop, "mid")
        lower_stat = self._evaluate_tier(lower_crop, "lower")

        # Detect mask or occlusion on lower face
        is_masked = False
        if lower_stat["mask_prob"] > 0.45 or (lower_stat["skin_ratio"] < 0.22 and lower_stat["edge_density"] > 0.08):
            is_masked = True

        # Check overall visibility
        avg_skin = (upper_stat["skin_ratio"] + mid_stat["skin_ratio"] + lower_stat["skin_ratio"]) / 3.0
        
        if avg_skin < 0.12 and upper_stat["skin_ratio"] < 0.10:
            face_status = "UNAVAILABLE" # Rear view or severe darkness
            face_visible = False
        elif is_masked:
            face_status = "MASKED_LOWER"
            face_visible = True
        elif upper_stat["visible"] and not lower_stat["visible"]:
            face_status = "PARTIAL_UPPER"
            face_visible = True
        else:
            face_status = "FULL_VISIBLE"
            face_visible = True

        # Generate face embedding from visible tiers
        embedding = self._compute_face_embedding(face_crop, upper_crop, mid_crop, lower_crop, is_masked)

        return {
            "is_detected": face_visible or detection_type == "CASCADE_MATCH",
            "face_visible": face_visible,
            "face_status": face_status,
            "detection_type": detection_type,
            "face_box": face_box,
            "tiers": {
                "upper": upper_stat,
                "mid": mid_stat,
                "lower": lower_stat,
            },
            "mask_detected": is_masked,
            "mask_confidence": round(lower_stat["mask_prob"], 2),
            "face_embedding": embedding
        }

    def _evaluate_tier(self, crop: np.ndarray, tier_name: str) -> Dict[str, Any]:
        """Evaluates visibility, skin presence, and mask texture of a tier."""
        if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return {"visible": False, "skin_ratio": 0.0, "mask_prob": 0.0, "edge_density": 0.0}

        # Convert to HSV and YCrCb for skin tone identification
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)

        # Standard skin tone range in HSV and YCrCb
        skin_mask_hsv = cv2.inRange(hsv, np.array([0, 20, 50], dtype=np.uint8), np.array([30, 255, 255], dtype=np.uint8))
        skin_mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
        combined_skin = cv2.bitwise_and(skin_mask_hsv, skin_mask_ycrcb)

        total_pixels = crop.shape[0] * crop.shape[1]
        skin_pixels = cv2.countNonZero(combined_skin)
        skin_ratio = float(skin_pixels) / max(1, total_pixels)

        # Edge analysis for mask fabric or folds
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(cv2.countNonZero(edges)) / max(1, total_pixels)

        # Mask probability (uniform surgical blue/white/black or fabric mask with low skin tone)
        # Check surgical mask hues (blue: H in [90, 125], or dark black/cloth)
        blue_mask = cv2.inRange(hsv, np.array([85, 40, 40], dtype=np.uint8), np.array([135, 255, 255], dtype=np.uint8))
        blue_ratio = float(cv2.countNonZero(blue_mask)) / max(1, total_pixels)

        mask_prob = 0.0
        if tier_name == "lower":
            if blue_ratio > 0.20:
                mask_prob = 0.85 + blue_ratio * 0.15
            elif skin_ratio < 0.20 and edge_density > 0.06:
                mask_prob = 0.65 + edge_density * 0.35
            elif skin_ratio < 0.15:
                mask_prob = 0.50
        mask_prob = min(1.0, max(0.0, mask_prob))

        visible = skin_ratio > 0.18 or (tier_name == "upper" and edge_density > 0.05)

        return {
            "visible": visible,
            "skin_ratio": round(skin_ratio, 3),
            "mask_prob": round(mask_prob, 3),
            "edge_density": round(edge_density, 3),
            "crop_shape": [crop.shape[0], crop.shape[1]]
        }

    def _compute_face_embedding(self, full_crop: np.ndarray, upper: np.ndarray, mid: np.ndarray, lower: np.ndarray, is_masked: bool) -> List[float]:
        """Generate normalized 128-d face descriptor with tier weighting (suppressing occluded lower tier)."""
        # Rescale crops to standardized sizes
        full_res = cv2.resize(full_crop, (32, 32)).flatten().astype(float)
        up_res = cv2.resize(upper, (16, 16)).flatten().astype(float)
        mid_res = cv2.resize(mid, (16, 16)).flatten().astype(float)
        
        # If masked, zero-out lower tier influence to keep embedding robust!
        if is_masked or lower is None or lower.size == 0:
            low_res = np.zeros(256, dtype=float)
            tier_weights = (0.50, 0.50, 0.0)
        else:
            low_res = cv2.resize(lower, (16, 16)).flatten().astype(float)
            tier_weights = (0.40, 0.35, 0.25)

        # Concatenate and project to 128-dimensional embedding
        raw_vec = np.concatenate([
            full_res[:32] * 0.2,
            up_res[:48] * tier_weights[0],
            mid_res[:32] * tier_weights[1],
            low_res[:16] * tier_weights[2]
        ])

        norm = np.linalg.norm(raw_vec)
        if norm > 0:
            raw_vec = raw_vec / norm

        # Pad or slice to exactly 128 floats
        emb = np.zeros(128, dtype=float)
        emb[:min(128, len(raw_vec))] = raw_vec[:min(128, len(raw_vec))]
        return [float(x) for x in emb]

    def _empty_response(self, reason: str) -> Dict[str, Any]:
        return {
            "is_detected": False,
            "face_visible": False,
            "face_status": "UNAVAILABLE",
            "detection_type": reason,
            "face_box": [0, 0, 0, 0],
            "tiers": {
                "upper": {"visible": False, "skin_ratio": 0.0, "mask_prob": 0.0},
                "mid": {"visible": False, "skin_ratio": 0.0, "mask_prob": 0.0},
                "lower": {"visible": False, "skin_ratio": 0.0, "mask_prob": 0.0},
            },
            "mask_detected": False,
            "mask_confidence": 0.0,
            "face_embedding": [0.0] * 128
        }
