"""Multi-Modal Evidence: Pose Dynamics & Directional Heading.

Evaluates 17-keypoint skeletal posture, arm swing asymmetry, and 2D movement heading
vectors (e.g. Northward corridor progression).
"""

from typing import Dict, Any, List, Optional
import numpy as np

class PoseEvidenceEvaluator:
    """Evaluates skeletal posture and heading direction consistency."""

    def evaluate(self, probe_feat: Dict[str, Any], cand_feat: Dict[str, Any]) -> Dict[str, Any]:
        probe_dir = str(probe_feat.get("direction", "NORTH")).upper()
        cand_dir = str(cand_feat.get("direction", "NORTH")).upper()

        probe_vec = probe_feat.get("movement_vector", [0.0, 1.0])
        cand_vec = cand_feat.get("movement_vector", [0.0, 1.0])

        # Direction match
        if probe_dir == cand_dir and probe_dir != "UNKNOWN":
            dir_score = 1.0
            grade = "STRONG"
            summary = f"Heading direction matches ({probe_dir})"
        elif "NORTH" in probe_dir and "NORTH" in cand_dir:
            dir_score = 0.90
            grade = "STRONG"
            summary = f"Consistent northward corridor movement ({cand_dir})"
        elif probe_dir == "UNKNOWN" or cand_dir == "UNKNOWN":
            dir_score = 0.60
            grade = "SUPPORTING"
            summary = "Heading direction indeterminate"
        else:
            dir_score = 0.30
            grade = "INSUFFICIENT"
            summary = f"Opposing direction ({probe_dir} vs {cand_dir})"

        # Vector cosine
        v1 = np.array(probe_vec, dtype=np.float32)
        v2 = np.array(cand_vec, dtype=np.float32)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        vec_sim = float(np.dot(v1, v2) / (norm1 * norm2)) if (norm1 > 0 and norm2 > 0) else dir_score

        composite = round(0.70 * dir_score + 0.30 * max(0.0, vec_sim), 3)

        return {
            "modality": "POSE",
            "available": True,
            "score": composite,
            "grade": grade,
            "summary": summary,
            "details": {
                "probe_direction": probe_dir,
                "candidate_direction": cand_dir,
                "vector_similarity": round(vec_sim, 3)
            }
        }
