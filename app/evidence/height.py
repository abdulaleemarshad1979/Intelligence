"""Multi-Modal Evidence: Calibrated Physical Stature & Height Estimation.

Evaluates ground-plane perspective projection stature estimates in centimeters
with calibrated camera tilt tolerance bounds (e.g., 178 cm ± 3.5 cm).
"""

from typing import Dict, Any, Optional

class HeightEvidenceEvaluator:
    """Evaluates estimated physical stature compatibility."""

    def evaluate(self, probe_feat: Dict[str, Any], cand_feat: Dict[str, Any]) -> Dict[str, Any]:
        probe_h = float(probe_feat.get("height_cm", 175.0))
        cand_h = float(cand_feat.get("height_cm", 175.0))

        if probe_h <= 0 or cand_h <= 0:
            return {
                "modality": "HEIGHT",
                "available": False,
                "score": 0.0,
                "grade": "UNAVAILABLE",
                "summary": "Height calibration unavailable",
                "details": {}
            }

        diff = abs(probe_h - cand_h)
        # Tolerance margin 4.0 cm
        if diff <= 4.0:
            score = 1.0 - (diff / 20.0)
            grade = "STRONG"
            summary = f"Height consistent ({cand_h:.1f} cm vs probe {probe_h:.1f} cm, diff {diff:.1f} cm)"
        elif diff <= 8.0:
            score = 1.0 - (diff / 15.0)
            grade = "SUPPORTING"
            summary = f"Height compatible ({cand_h:.1f} cm vs probe {probe_h:.1f} cm, diff {diff:.1f} cm)"
        elif diff <= 12.0:
            score = max(0.2, 1.0 - (diff / 12.0))
            grade = "MODERATE"
            summary = f"Height variance ({cand_h:.1f} cm vs {probe_h:.1f} cm)"
        else:
            score = 0.0
            grade = "INSUFFICIENT"
            summary = f"Height mismatch ({cand_h:.1f} cm vs {probe_h:.1f} cm, diff {diff:.1f} cm)"

        return {
            "modality": "HEIGHT",
            "available": True,
            "score": round(max(0.0, score), 3),
            "grade": grade,
            "summary": summary,
            "details": {
                "probe_height_cm": probe_h,
                "candidate_height_cm": cand_h,
                "height_diff_cm": round(diff, 1)
            }
        }
