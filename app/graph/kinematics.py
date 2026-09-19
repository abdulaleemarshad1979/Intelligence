"""Kinematic Validation & Transition Probability Engine.

Enforces strict physical human locomotion bounds (vmin = 0.5 m/s to vmax = 3.0 m/s)
along the physical pedestrian street network, pruning physically impossible visual matches
across disconnected city sectors and calculating calibrated Gaussian transition probabilities.
"""

import math
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class KinematicValidator:
    """Validates pedestrian transit feasibility between camera nodes."""

    def __init__(
        self,
        camera_distances: Optional[Dict[str, Dict[str, float]]] = None,
        v_min: float = 0.5,
        v_max: float = 3.0,
        v_nominal: float = 1.34
    ):
        """
        Args:
            camera_distances: Nested dict [from_cam][to_cam] -> street distance in meters.
            v_min: Minimum locomotion velocity (0.5 m/s - slow walk / elderly).
            v_max: Maximum locomotion velocity (3.0 m/s - fast sprint).
            v_nominal: Average pedestrian walking speed (1.34 m/s).
        """
        self.camera_distances: Dict[str, Dict[str, float]] = camera_distances or {}
        self.v_min = v_min
        self.v_max = v_max
        self.v_nominal = v_nominal

    def set_distance(self, from_cam: str, to_cam: str, distance_m: float, bidirectional: bool = True):
        if from_cam not in self.camera_distances:
            self.camera_distances[from_cam] = {}
        self.camera_distances[from_cam][to_cam] = float(distance_m)

        if bidirectional:
            if to_cam not in self.camera_distances:
                self.camera_distances[to_cam] = {}
            self.camera_distances[to_cam][from_cam] = float(distance_m)

    def get_distance(self, from_cam: str, to_cam: str) -> Optional[float]:
        if from_cam == to_cam:
            return 0.0
        return self.camera_distances.get(from_cam, {}).get(to_cam, None)

    def is_kinematically_feasible(self, from_cam: str, to_cam: str, delta_t: float) -> bool:
        """Determines whether a transit time delta_t (seconds) conforms to human locomotion bounds."""
        if from_cam == to_cam:
            return delta_t >= 0.0

        dist = self.get_distance(from_cam, to_cam)
        if dist is None or delta_t <= 0.0:
            return False

        t_min = dist / self.v_max
        t_max = dist / self.v_min
        return t_min <= delta_t <= t_max

    def compute_transition_probability(
        self,
        from_cam: str,
        to_cam: str,
        delta_t: float
    ) -> float:
        """Calculates calibrated Gaussian spatio-temporal transition probability P_transition.

        Returns:
            P_transition in [0, 1]. Returns 0.0 if kinematically impossible (I = 0).
        """
        if from_cam == to_cam:
            return 1.0 if delta_t >= 0.0 else 0.0

        dist = self.get_distance(from_cam, to_cam)
        if dist is None or delta_t <= 0.0:
            return 0.0

        t_min = dist / self.v_max
        t_max = dist / self.v_min

        # Indicator function strict pruning
        if not (t_min <= delta_t <= t_max):
            return 0.0

        # Calibrate mean travel time and variance based on street distance
        mu = dist / self.v_nominal
        sigma = max(1.5, 0.25 * mu)

        exponent = -((delta_t - mu) ** 2) / (2.0 * (sigma ** 2))
        prob = math.exp(exponent)
        return float(max(0.001, min(1.0, prob)))
