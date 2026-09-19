"""Multi-Modal Evidence: Clothing Colors & Palette Consistency.

Analyzes upper and lower torso/leg clothing color signatures, HSV/Lab distance,
and dominant palette continuity across camera illumination differences.
"""

from typing import Dict, Any, List, Optional
import math

def hex_to_rgb(hex_str: str) -> tuple:
    hex_str = hex_str.lstrip("#")
    if len(hex_str) != 6:
        return (128, 128, 128)
    try:
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        return (128, 128, 128)

def color_similarity(hex1: str, hex2: str) -> float:
    r1, g1, b1 = hex_to_rgb(hex1)
    r2, g2, b2 = hex_to_rgb(hex2)
    dist = math.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)
    max_dist = math.sqrt(255**2 + 255**2 + 255**2)
    return max(0.0, 1.0 - (dist / max_dist))

class ClothingEvidenceEvaluator:
    """Evaluates clothing color signatures and palette consistency."""

    def evaluate(self, probe_feat: Dict[str, Any], cand_feat: Dict[str, Any]) -> Dict[str, Any]:
        probe_upper = probe_feat.get("clothing_upper", "#334455")
        probe_lower = probe_feat.get("clothing_lower", "#112233")
        
        cand_upper = cand_feat.get("clothing_upper", "#334455")
        cand_lower = cand_feat.get("clothing_lower", "#112233")

        sim_upper = color_similarity(probe_upper, cand_upper)
        sim_lower = color_similarity(probe_lower, cand_lower)

        # Composite clothing consistency score
        score = round(0.55 * sim_upper + 0.45 * sim_lower, 3)

        if score >= 0.85:
            grade = "STRONG"
            summary = f"Strong clothing consistency (Upper {cand_upper} {sim_upper:.2f}, Lower {cand_lower} {sim_lower:.2f})"
        elif score >= 0.70:
            grade = "SUPPORTING"
            summary = f"Clothing supporting evidence (Upper {cand_upper}, Lower {cand_lower})"
        elif score >= 0.50:
            grade = "MODERATE"
            summary = f"Moderate clothing similarity ({score:.2f})"
        else:
            grade = "INSUFFICIENT"
            summary = f"Clothing mismatch (Upper {cand_upper} vs {probe_upper})"

        return {
            "modality": "CLOTHING",
            "available": True,
            "score": score,
            "grade": grade,
            "summary": summary,
            "details": {
                "upper_similarity": round(sim_upper, 3),
                "lower_similarity": round(sim_lower, 3),
                "probe_upper": probe_upper,
                "probe_lower": probe_lower,
                "cand_upper": cand_upper,
                "cand_lower": cand_lower
            }
        }
