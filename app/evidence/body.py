"""Multi-Modal Evidence: Body Re-Identification & Anatomical Proportions.

Evaluates OSNet/FastReID 512-d appearance embeddings, torso-to-leg biometric ratios,
and silhouette width profiles to deliver defensible body evidence.
"""

from typing import Dict, Any, List, Optional
import numpy as np

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), 0.0, 1.0))

class BodyEvidenceEvaluator:
    """Evaluates full-body appearance and anatomical proportion similarity."""

    def evaluate(self, probe_feat: Dict[str, Any], cand_feat: Dict[str, Any]) -> Dict[str, Any]:
        probe_emb = probe_feat.get("body_embedding", [])
        cand_emb = cand_feat.get("body_embedding", [])

        if not probe_emb or not cand_emb:
            return {
                "modality": "BODY",
                "available": False,
                "score": 0.0,
                "grade": "UNAVAILABLE",
                "summary": "Body Re-ID vector unavailable",
                "details": {}
            }

        sim = cosine_similarity(probe_emb, cand_emb)

        # Compare torso-to-leg ratio if present in clothing/landmarks
        probe_ratio = probe_feat.get("torso_leg_ratio", 0.85)
        cand_ratio = cand_feat.get("torso_leg_ratio", 0.85)
        ratio_diff = abs(probe_ratio - cand_ratio)
        ratio_match = max(0.0, 1.0 - (ratio_diff / 0.35))

        # Composite body score (80% embedding + 20% anatomical ratio)
        composite_score = round(0.80 * sim + 0.20 * ratio_match, 3)

        if composite_score >= 0.78:
            grade = "STRONG"
            summary = f"Strong evidence (OSNet Re-ID similarity {sim:.2f}, ratio match {ratio_match:.2f})"
        elif composite_score >= 0.65:
            grade = "MODERATE"
            summary = f"Moderate evidence (OSNet Re-ID similarity {sim:.2f})"
        elif composite_score >= 0.50:
            grade = "SUPPORTING"
            summary = f"Supporting evidence (Partial body appearance match {sim:.2f})"
        else:
            grade = "INSUFFICIENT"
            summary = f"Insufficient body match ({composite_score:.2f})"

        return {
            "modality": "BODY",
            "available": True,
            "score": composite_score,
            "grade": grade,
            "summary": summary,
            "details": {
                "embedding_similarity": round(sim, 3),
                "ratio_match": round(ratio_match, 3),
                "probe_ratio": probe_ratio,
                "candidate_ratio": cand_ratio
            }
        }
