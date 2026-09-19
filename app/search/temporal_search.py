"""Temporal Search & Velocity Feasibility Engine.

Filters candidate observations using temporal windows and human physical movement
feasibility models (walking velocity 0.5 - 2.5 m/s, sprint velocity up to 6.0 m/s).
"""

from typing import Dict, Any, List, Optional, Tuple

class TemporalSearchEngine:
    """Manages time-window searches and velocity constraint validations."""

    def __init__(self, min_speed_mps: float = 0.4, max_speed_mps: float = 6.0):
        self.min_speed_mps = min_speed_mps
        self.max_speed_mps = max_speed_mps

    def check_transit_feasibility(
        self,
        t_start: float,
        t_end: float,
        distance_meters: float
    ) -> Tuple[bool, float, str]:
        delta_t = t_end - t_start
        if delta_t <= 0:
            return False, 0.0, "Negative or simultaneous transit time"

        speed = distance_meters / delta_t
        if speed > self.max_speed_mps:
            return False, speed, f"Speed {speed:.1f} m/s exceeds human running capacity ({self.max_speed_mps} m/s)"
        if speed < self.min_speed_mps:
            return True, speed, f"Slow or loitering transit ({speed:.2f} m/s)"
        return True, speed, f"Realistic pedestrian pace ({speed:.2f} m/s)"

    def filter_by_time_window(
        self,
        observations: List[Dict[str, Any]],
        t_start: float,
        t_end: float
    ) -> List[Dict[str, Any]]:
        return [
            obs for obs in observations
            if t_start <= (obs.get("timestamp") or obs.get("start_time") or 0.0) <= t_end
        ]
