"""Base interfaces for Detection adapters."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from app.adapters.base import DetectionResult, BaseDetector


class BasePedestrianDetector(BaseDetector, ABC):
    """Abstract base class for pedestrian detectors."""
    pass
