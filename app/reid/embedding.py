import json
import numpy as np
from typing import List, Any

def cosine_similarity(v1: Any, v2: Any) -> float:
    if not v1 or not v2:
        return 0.0
    if isinstance(v1, str):
        try:
            v1 = json.loads(v1)
        except Exception:
            return 0.0
    if isinstance(v2, str):
        try:
            v2 = json.loads(v2)
        except Exception:
            return 0.0
    if len(v1) == 0 or len(v2) == 0:
        return 0.0

    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    if len(a) != len(b):
        min_dim = min(len(a), len(b))
        a = a[:min_dim]
        b = b[:min_dim]
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    sim = float(np.dot(a, b) / (norm_a * norm_b))
    return max(0.0, min(1.0, sim))

def euclidean_distance(v1: Any, v2: Any) -> float:
    if not v1 or not v2:
        return 0.0
    if isinstance(v1, str):
        try:
            v1 = json.loads(v1)
        except Exception:
            return 0.0
    if isinstance(v2, str):
        try:
            v2 = json.loads(v2)
        except Exception:
            return 0.0
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    if len(a) != len(b):
        min_dim = min(len(a), len(b))
        a = a[:min_dim]
        b = b[:min_dim]
    return float(np.linalg.norm(a - b))
