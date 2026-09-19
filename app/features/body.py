"""Body biometrics, proportions, torso/leg length, and clothing color analysis."""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, List

def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"

class BodyAnalyzer:
    def __init__(self):
        pass

    def analyze(self, person_crop: np.ndarray) -> Dict[str, Any]:
        """Extract body lengths, proportions, clothing colors, and appearance embedding."""
        if person_crop is None or person_crop.size == 0:
            return self._empty_response()

        ph, pw = person_crop.shape[:2]
        if ph < 40 or pw < 15:
            return self._empty_response()

        # Approximate anatomical landmarks relative to bounding box
        # Neck line ~ 16% ph
        # Waist line ~ 54% ph
        # Ankle / foot base ~ 98% ph
        neck_y = int(ph * 0.16)
        waist_y = int(ph * 0.54)
        foot_y = int(ph * 0.98)

        torso_len = max(1, waist_y - neck_y)
        leg_len = max(1, foot_y - waist_y)
        total_body_len = max(1, foot_y - neck_y)

        torso_leg_ratio = round(float(torso_len) / float(leg_len), 3)

        # Silhouette width measurements
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Shoulder line width (sampled between 18% and 24% ph)
        shoulder_band = thresh[int(ph * 0.18):int(ph * 0.24), :]
        shoulder_width = int(np.mean(np.sum(shoulder_band > 0, axis=1))) if shoulder_band.size > 0 else int(pw * 0.75)
        shoulder_width = min(pw, max(10, shoulder_width))

        # Waist line width
        waist_band = thresh[int(ph * 0.50):int(ph * 0.56), :]
        waist_width = int(np.mean(np.sum(waist_band > 0, axis=1))) if waist_band.size > 0 else int(pw * 0.60)
        waist_width = min(pw, max(8, waist_width))

        # Clothing Color Extraction
        # Upper clothing: between 20% and 52% ph
        upper_crop = person_crop[int(ph * 0.20):int(ph * 0.52), int(pw * 0.15):int(pw * 0.85)]
        upper_hex, upper_rgb = self._extract_dominant_color(upper_crop, default="#3b4252")

        # Lower clothing: between 56% and 92% ph
        lower_crop = person_crop[int(ph * 0.56):int(ph * 0.92), int(pw * 0.15):int(pw * 0.85)]
        lower_hex, lower_rgb = self._extract_dominant_color(lower_crop, default="#1e222a")

        # Generate 256-d appearance embedding (Spatial Pyramid HSV Color Histogram + Aspect Ratio)
        body_embedding = self._compute_appearance_embedding(person_crop, upper_crop, lower_crop, torso_leg_ratio)

        return {
            "body_length_px": total_body_len,
            "torso_length_px": torso_len,
            "leg_length_px": leg_len,
            "torso_leg_ratio": torso_leg_ratio,
            "shoulder_width_px": shoulder_width,
            "waist_width_px": waist_width,
            "proportions": {
                "torso_leg_ratio": torso_leg_ratio,
                "torso_pct": round(torso_len / float(total_body_len), 3),
                "leg_pct": round(leg_len / float(total_body_len), 3),
                "shoulder_waist_ratio": round(shoulder_width / max(1.0, float(waist_width)), 2),
                "aspect_ratio": round(ph / max(1.0, float(pw)), 2)
            },
            "clothing": {
                "upper_hex": upper_hex,
                "upper_rgb": upper_rgb,
                "lower_hex": lower_hex,
                "lower_rgb": lower_rgb
            },
            "body_embedding": body_embedding
        }

    def _extract_dominant_color(self, crop: np.ndarray, default: str) -> Tuple[str, List[int]]:
        if crop is None or crop.size == 0:
            return default, [50, 60, 70]
        # Rescale crop for fast K-means
        small = cv2.resize(crop, (24, 24)).reshape(-1, 3).astype(np.float32)
        # Use median of the central distribution to resist background noise
        median_bgr = np.median(small, axis=0)
        r = int(np.clip(median_bgr[2], 0, 255))
        g = int(np.clip(median_bgr[1], 0, 255))
        b = int(np.clip(median_bgr[0], 0, 255))
        return rgb_to_hex(r, g, b), [r, g, b]

    def _compute_appearance_embedding(self, person_crop: np.ndarray, upper: np.ndarray, lower: np.ndarray, ratio: float) -> List[float]:
        hsv_full = cv2.cvtColor(cv2.resize(person_crop, (32, 64)), cv2.COLOR_BGR2HSV)
        
        # 3D HSV histogram: 8 Hue bins, 4 Sat bins, 4 Val bins = 128 bins
        hist_full = cv2.calcHist([hsv_full], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
        cv2.normalize(hist_full, hist_full)

        # Upper region histogram: 4 Hue, 4 Sat, 4 Val = 64 bins
        if upper.size > 0:
            hsv_up = cv2.cvtColor(cv2.resize(upper, (16, 32)), cv2.COLOR_BGR2HSV)
            hist_up = cv2.calcHist([hsv_up], [0, 1, 2], None, [4, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
            cv2.normalize(hist_up, hist_up)
        else:
            hist_up = np.zeros(64, dtype=float)

        # Lower region histogram: 4 Hue, 4 Sat, 4 Val = 64 bins
        if lower.size > 0:
            hsv_low = cv2.cvtColor(cv2.resize(lower, (16, 32)), cv2.COLOR_BGR2HSV)
            hist_low = cv2.calcHist([hsv_low], [0, 1, 2], None, [4, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
            cv2.normalize(hist_low, hist_low)
        else:
            hist_low = np.zeros(64, dtype=float)

        combined = np.concatenate([hist_full, hist_up, hist_low])
        # Inject ratio signal into embedding
        combined[0] += ratio * 0.1
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        emb = np.zeros(256, dtype=float)
        emb[:min(256, len(combined))] = combined[:min(256, len(combined))]
        return [float(x) for x in emb]

    def _empty_response(self) -> Dict[str, Any]:
        return {
            "body_length_px": 0,
            "torso_length_px": 0,
            "leg_length_px": 0,
            "torso_leg_ratio": 0.85,
            "shoulder_width_px": 0,
            "waist_width_px": 0,
            "proportions": {"torso_pct": 0.5, "leg_pct": 0.5, "shoulder_waist_ratio": 1.2, "aspect_ratio": 2.5},
            "clothing": {"upper_hex": "#000000", "upper_rgb": [0,0,0], "lower_hex": "#000000", "lower_rgb": [0,0,0]},
            "body_embedding": [0.0] * 256
        }
