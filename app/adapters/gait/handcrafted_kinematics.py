"""Handcrafted Kinematics Gait Adapter (Proprietary IP - Approved).

Derived from:
- Joint angle velocities (knee and hip angular velocity d(theta)/dt)
- Stride frequency (cadence in Hz)
- FFT harmonic ratios and spectral distribution
Zero 3rd-party licensing risk.
"""

from typing import List, Dict, Any, Optional
from app.adapters.gait.base import GaitAdapter
from app.features.gait import GaitAnalyzer


class HandcraftedKinematicsAdapter(GaitAdapter):
    """Production Approved Handcrafted Kinematics Adapter."""

    def __init__(self, sequence_length: int = 24, fps: float = 25.0):
        self.sequence_length = sequence_length
        self.fps = fps
        self.analyzer = GaitAnalyzer(window_size=sequence_length, fps=fps)
        self.backend = "HANDCRAFTED_KINEMATICS_ENGINE (Proprietary IP - Approved)"

    def extract_sequence(self, track_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid_frames = []
        for f in track_history:
            if "inter_ankle_dist" in f or "keypoints_crop" in f or "bbox" in f:
                valid_frames.append(f)
        return valid_frames[-self.sequence_length:]

    def quality_score(self, sequence: List[Dict[str, Any]]) -> float:
        if not sequence:
            return 0.0
        return min(1.0, len(sequence) / max(1, self.sequence_length))

    def generate_embedding(self, sequence: List[Dict[str, Any]], estimated_height_cm: float = 170.0) -> List[float]:
        res = self.analyzer.analyze_sequence(sequence, estimated_height_cm=estimated_height_cm)
        return res.get("gait_embedding", [0.0] * 64)

    def analyze_sequence(self, track_history: List[Dict[str, Any]], estimated_height_cm: float = 170.0) -> Dict[str, Any]:
        seq = self.extract_sequence(track_history)
        return self.analyzer.analyze_sequence(seq, estimated_height_cm=estimated_height_cm)

    def get_backend_info(self) -> Dict[str, Any]:
        return {
            "model_id": "handcrafted_kinematics",
            "name": "Handcrafted Kinematics",
            "backend": self.backend,
            "production_status": "Approved",
            "license": "Proprietary IP",
            "licensing_risk": "Zero 3rd-party licensing risk",
            "feature_dim": 64,
            "mathematical_basis": "Derived from joint angle velocities, stride frequency, and FFT harmonic ratios."
        }
