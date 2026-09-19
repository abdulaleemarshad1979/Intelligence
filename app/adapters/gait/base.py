"""Base interfaces for Gait Recognition and Temporal Dynamics adapters.

Defines the pluggable GaitAdapter interface supporting OpenGait (GaitBase, GaitGL, GaitSet)
and specialized kinematic dynamics models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np
from app.adapters.base import BaseGaitModel, GaitSequenceResult


class GaitAdapter(BaseGaitModel, ABC):
    """Unified adapter interface for gait recognition architectures."""

    @abstractmethod
    def extract_sequence(self, track_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and extract valid sequential observations (silhouettes, ankles, keypoints)."""
        pass

    @abstractmethod
    def generate_embedding(self, sequence: List[Dict[str, Any]], estimated_height_cm: float = 170.0) -> List[float]:
        """Generate normalized cross-view gait feature embedding vector."""
        pass

    @abstractmethod
    def quality_score(self, sequence: List[Dict[str, Any]]) -> float:
        """Compute the temporal quality and completeness score for the sequence."""
        pass

    def analyze_sequence(self, track_history: List[Dict[str, Any]], estimated_height_cm: float = 170.0) -> Dict[str, Any]:
        """Default orchestration combining extraction, embedding, kinematics, and quality scoring."""
        seq = self.extract_sequence(track_history)
        q = self.quality_score(seq)
        emb = self.generate_embedding(seq, estimated_height_cm=estimated_height_cm)
        return {
            "gait_embedding": emb,
            "quality_score": q,
            "sequence_length": len(seq),
            "model_backend": self.get_backend_info().get("backend", "GAIT_ADAPTER")
        }
