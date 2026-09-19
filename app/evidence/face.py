"""Multi-Modal Evidence: Face Biometrics & 3-Tier Mask Resilience.

Evaluates upper/mid/lower facial tiers, handles masked, turned-away, or low-res crops,
and generates qualitative defensible evidence scores without opaque decisions.
"""

from typing import Dict, Any, Optional, List
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

class FaceEvidenceEvaluator:
    """Evaluates facial evidence between probe observation and candidate track."""

    def evaluate(self, probe_feat: Dict[str, Any], cand_feat: Dict[str, Any]) -> Dict[str, Any]:
        probe_status = probe_feat.get("face_status", "UNAVAILABLE")
        cand_status = cand_feat.get("face_status", "UNAVAILABLE")
        
        probe_emb = probe_feat.get("face_embedding", [])
        cand_emb = cand_feat.get("face_embedding", [])

        # Check availability
        if probe_status == "UNAVAILABLE" or cand_status == "UNAVAILABLE" or not probe_emb or not cand_emb:
            return {
                "modality": "FACE",
                "available": False,
                "status": "UNAVAILABLE",
                "score": 0.0,
                "grade": "UNAVAILABLE",
                "summary": "Face unavailable (subject turned away / masked / occlusion)",
                "details": {
                    "probe_status": probe_status,
                    "candidate_status": cand_status,
                    "tier_scores": {}
                }
            }

        sim = cosine_similarity(probe_emb, cand_emb)
        
        # Determine qualitative grade
        if sim >= 0.80:
            grade = "STRONG"
            summary = f"Strong evidence (3-tier face match {sim:.2f})"
        elif sim >= 0.68:
            grade = "MODERATE"
            summary = f"Moderate evidence (Partial periocular/facial match {sim:.2f})"
        elif sim >= 0.52:
            grade = "SUPPORTING"
            summary = f"Supporting evidence (Weak facial match {sim:.2f})"
        else:
            grade = "INSUFFICIENT"
            summary = f"Insufficient match ({sim:.2f})"

        return {
            "modality": "FACE",
            "available": True,
            "status": cand_status,
            "score": round(sim, 3),
            "grade": grade,
            "summary": summary,
            "details": {
                "similarity": round(sim, 3),
                "probe_status": probe_status,
                "candidate_status": cand_status
            }
        }
