"""CCTV Motion Semantics & Activity Profiling.

Computes motion vectors, directional compass orientation (N, NE, E, SE, S, SW, W, NW),
transit speed profiling, and dwell-time / loitering detection.
"""

from typing import Dict, Any, List, Tuple
import math

class MotionSemanticsAnalyzer:
    """Analyzes spatio-temporal trajectories for behavioral motion cues."""

    def compute_heading(self, dx: float, dy: float) -> str:
        """Compute 8-point compass heading from movement vector (dx, dy)."""
        if abs(dx) < 1.0 and abs(dy) < 1.0:
            return "STATIONARY"

        # In screen coordinates, y increases downwards, so north is negative dy
        angle_rad = math.atan2(-dy, dx)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360

        compass_sectors = [
            ("EAST", 22.5, 67.5),
            ("NORTH_EAST", 67.5, 112.5),
            ("NORTH", 112.5, 157.5),
            ("NORTH_WEST", 157.5, 202.5),
            ("WEST", 202.5, 247.5),
            ("SOUTH_WEST", 247.5, 292.5),
            ("SOUTH", 292.5, 337.5)
        ]

        if angle_deg >= 337.5 or angle_deg < 22.5:
            return "EAST"

        for sector, low, high in compass_sectors:
            if low <= angle_deg < high:
                return sector

        return "NORTH"
