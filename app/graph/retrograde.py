"""Spatio-Temporal Retrograde Trajectory Reconstruction Engine.

Executes reverse temporal traversal across the municipal camera network to trace a suspect's
spatial path backward in time from a crime scene locus back to their point of origin, ingress vehicle,
or unmonitored blind zone, enforcing kinematic human locomotion reachability bounds.
"""

import math
import logging
from typing import List, Dict, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class RetrogradeTracker:
    """Retrograde backward-path trajectory reconstruction engine adhering to Section 8 C."""

    def __init__(
        self,
        camera_distances: Dict[str, Dict[str, float]],
        v_min: float = 0.5,
        v_max: float = 3.0,
        score_threshold: float = 0.70
    ):
        """
        Args:
            camera_distances: camera_distances[cam_i][cam_j] = physical distance in meters.
            v_min: minimum locomotion speed (0.5 m/s).
            v_max: maximum locomotion speed (3.0 m/s).
            score_threshold: minimum composite similarity for backward association (default 0.70).
        """
        self.dist_matrix = camera_distances
        self.v_min = v_min
        self.v_max = v_max
        self.score_threshold = score_threshold

    def is_kinematically_feasible(self, from_cam: str, to_cam: str, delta_t: float) -> bool:
        """Determines whether a transit from from_cam to to_cam in delta_t seconds is physically feasible."""
        if from_cam == to_cam:
            return delta_t >= 0.0

        dist = self.dist_matrix.get(from_cam, {}).get(to_cam, None)
        if dist is None or delta_t <= 0.0:
            return False

        min_time = dist / self.v_max
        max_time = dist / self.v_min
        return min_time <= delta_t <= max_time

    def find_backward_origin(self, incident_probe: Dict[str, Any], candidate_pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Iteratively steps backward from the incident scene tracklet to isolate point of origin.

        Args:
            incident_probe: dict containing camera_id, t_in, t_out, embedding (512-dim), active_attributes.
            candidate_pool: pool of candidate tracklet dicts across historical cameras.

        Returns:
            Ordered list of tracklets starting from incident probe and stepping backward in time to the origin.
        """
        reconstructed_trajectory = [incident_probe]
        current_anchor = incident_probe
        pool = list(candidate_pool)  # Work on local copy to mutate safely

        while True:
            best_predecessor: Optional[Dict[str, Any]] = None
            best_score = -1.0

            for cand in pool:
                # Must exit before anchor arrival (strict causality for backward search)
                delta_t = current_anchor["t_in"] - cand["t_out"]
                if not self.is_kinematically_feasible(cand["camera_id"], current_anchor["camera_id"], delta_t):
                    continue

                # Cosine similarity between 512-dim embeddings
                emb1 = np.array(current_anchor["embedding"], dtype=np.float32)
                emb2 = np.array(cand["embedding"], dtype=np.float32)
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                if norm1 < 1e-6 or norm2 < 1e-6:
                    sim_reid = 0.0
                else:
                    sim_reid = float(np.dot(emb1, emb2) / (norm1 * norm2))

                # Jaccard attribute similarity
                mask1 = set(current_anchor.get("active_attributes", []))
                mask2 = set(cand.get("active_attributes", []))
                union_len = len(mask1.union(mask2))
                sim_attr = len(mask1.intersection(mask2)) / max(union_len, 1) if union_len > 0 else 1.0

                composite_score = (0.65 * sim_reid) + (0.35 * sim_attr)
                if composite_score > best_score and composite_score >= self.score_threshold:
                    best_score = composite_score
                    best_predecessor = cand

            if best_predecessor is not None:
                reconstructed_trajectory.append(best_predecessor)
                pool.remove(best_predecessor)
                current_anchor = best_predecessor
            else:
                # No physically feasible predecessor found: origin reached or blind zone
                break

        return reconstructed_trajectory
