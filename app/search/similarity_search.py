"""Multi-Attribute & Embedding Similarity Search.

Executes vector cosine similarity and multi-attribute matching across CCTV feature stores.
"""

from typing import List, Dict, Any, Tuple
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

class SimilaritySearchEngine:
    """Performs feature ranking across candidate tracks."""

    def rank_candidates(
        self,
        probe_feature: Dict[str, Any],
        candidate_features: List[Dict[str, Any]],
        min_body_sim: float = 0.50
    ) -> List[Tuple[Dict[str, Any], float]]:
        ranked = []
        probe_body = probe_feature.get("body_embedding", [])

        for cand in candidate_features:
            cand_body = cand.get("body_embedding", [])
            body_sim = cosine_similarity(probe_body, cand_body) if (probe_body and cand_body) else 0.50

            if body_sim >= min_body_sim:
                ranked.append((cand, body_sim))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
