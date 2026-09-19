"""Dynamic Evidence Fusion Engine.

Combines multimodal evidence (Face, Body, Gait, Pose, Clothing, Height, Carried Objects)
with missing-modality resilience and dynamic weight normalization.
"""

from typing import Dict, Any

class EvidenceFusionEngine:
    """Fuses multimodal evidence components dynamically without penalizing unobservable modalities."""

    DEFAULT_BASE_WEIGHTS = {
        "face": 0.35,
        "body": 0.25,
        "gait": 0.15,
        "clothing": 0.15,
        "height": 0.05,
        "carried_objects": 0.05
    }

    def fuse(self, evidence_breakdown: Dict[str, Any]) -> Dict[str, Any]:
        active_weights = {}
        weighted_sum = 0.0
        available_count = 0

        for mod, base_w in self.DEFAULT_BASE_WEIGHTS.items():
            ev = evidence_breakdown.get(mod, {})
            if ev.get("available", False):
                active_weights[mod] = base_w
                available_count += 1

        total_w = sum(active_weights.values())
        if total_w > 0:
            for mod in active_weights:
                norm_w = active_weights[mod] / total_w
                score = evidence_breakdown[mod].get("score", 0.0)
                weighted_sum += norm_w * score
        else:
            weighted_sum = 0.0

        composite_score = round(weighted_sum, 3)

        return {
            "composite_score": composite_score,
            "available_modalities_count": available_count,
            "active_weights": {k: round(v / total_w, 2) for k, v in active_weights.items()} if total_w > 0 else {},
            "face_available": evidence_breakdown.get("face", {}).get("available", False),
            "breakdown": evidence_breakdown
        }
