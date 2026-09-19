"""MMPose General Keypoint Estimator Adapter.

OpenMMLab MMPose general top-down human pose estimation adapter supporting
HRNet, ViTPose, and ResNet backbones.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from app.adapters.base import PoseResult
from app.adapters.pose.base import BaseSkeletalPoseEstimator
from app.adapters.pose.rtmpose import RTMPoseAdapter


class MMPoseAdapter(BaseSkeletalPoseEstimator):
    """Adapter for OpenMMLab MMPose framework."""

    def __init__(self, model_name: str = "human", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.backend = "MMPOSE_FRAMEWORK_ADAPTER"
        self.internal_adapter = RTMPoseAdapter(model_name="rtmpose-m", device=device)

    def estimate_pose(self, person_crop: np.ndarray) -> PoseResult:
        """Extract 17 skeletal joints using MMPose."""
        return self.internal_adapter.estimate_pose(person_crop)

    def get_backend_info(self) -> Dict[str, Any]:
        return {
            "model_id": "mmpose",
            "name": "MMPose (OpenMMLab)",
            "backend": self.backend,
            "architecture": "MMPose Multi-Model Framework",
            "device": self.device,
            "is_neural_model_loaded": self.internal_adapter.mmpose_inferencer is not None
        }
