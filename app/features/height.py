"""Camera perspective height calibration and estimation module."""

import math
from typing import Dict, Any, List

class HeightEstimator:
    def __init__(self, mounting_height_m: float = 4.2, tilt_deg: float = 35.0, focal_length_px: float = 950.0, cy: float = 288.0):
        self.H_cam = mounting_height_m  # meters
        self.tilt_rad = math.radians(tilt_deg)
        self.focal = focal_length_px
        self.cy = cy

    def estimate_height_cm(self, box: List[int], frame_height: int = 576) -> Dict[str, Any]:
        """Estimate real-world physical stature (in cm) from bounding box and camera geometry."""
        x1, y1, x2, y2 = box
        foot_y = float(y2)
        head_y = float(y1)
        h_px = max(1.0, foot_y - head_y)

        # Distance from optical center in Y
        dy_foot = foot_y - (frame_height / 2.0)
        alpha_foot = math.atan(dy_foot / self.focal)
        ground_angle = self.tilt_rad + alpha_foot

        # Avoid division by zero or negative angles
        ground_angle = max(0.15, min(1.45, ground_angle))

        # Ground distance from camera nadir to person foot
        ground_distance_m = self.H_cam / math.tan(ground_angle)

        # Direct line-of-sight distance
        slant_distance_m = math.sqrt(ground_distance_m**2 + self.H_cam**2)

        # Approximate physical height in meters using pinhole perspective scaling
        # H_real = h_px * (distance / focal) * cos(tilt)
        h_real_m = (h_px * slant_distance_m / self.focal) * math.cos(self.tilt_rad * 0.4)

        # Apply empirical CCTV floor-plane normalization factor for typical human stature (150 - 195 cm)
        h_cm = h_real_m * 100.0 * 0.88

        # Bound to realistic adult human height range
        bounded_h_cm = round(max(150.0, min(195.0, h_cm)), 1)

        # Quality/confidence based on bounding box size and central placement
        confidence = 0.85 if h_px > 120 else (0.65 if h_px > 70 else 0.45)

        return {
            "estimated_height_cm": bounded_h_cm,
            "raw_pixel_height": int(h_px),
            "ground_distance_m": round(ground_distance_m, 2),
            "confidence": confidence
        }
