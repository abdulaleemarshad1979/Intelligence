"""Base interfaces for Pose Estimation adapters."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from app.adapters.base import BasePoseEstimator, PoseResult


class BaseSkeletalPoseEstimator(BasePoseEstimator, ABC):
    """Abstract base class for 17-keypoint skeletal pose estimation."""
    pass
