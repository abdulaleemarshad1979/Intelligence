"""Partial Face module for handling masked, occluded, or low-resolution facial features."""

import numpy as np
from typing import Dict, Any, List

def compute_partial_face_similarity(emb1: List[float], emb2: List[float], mask1: bool, mask2: bool) -> float:
    """Compute cosine similarity between two face embeddings, accounting for mask flags.
    
    If one or both individuals have masked lower faces, the similarity focuses on the
    first 80 dimensions (forehead, brow, ocular, and nose bridge).
    """
    if not emb1 or not emb2 or len(emb1) == 0 or len(emb2) == 0:
        return 0.0

    v1 = np.array(emb1, dtype=float)
    v2 = np.array(emb2, dtype=float)

    # If all zeros (unavailable)
    if np.all(v1 == 0) or np.all(v2 == 0):
        return 0.0

    # Ensure aligned dimension lengths
    if len(v1) != len(v2):
        min_dim = min(len(v1), len(v2))
        v1 = v1[:min_dim]
        v2 = v2[:min_dim]

    if mask1 or mask2:
        # Focus on upper & mid face dimensions (up to 80 or length)
        k = min(80, len(v1))
        v1_sub = v1[:k]
        v2_sub = v2[:k]
        n1 = np.linalg.norm(v1_sub)
        n2 = np.linalg.norm(v2_sub)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v1_sub, v2_sub) / (n1 * n2))
    else:
        # Full face comparison
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))
