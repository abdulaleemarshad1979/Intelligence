"""Base interfaces for Face Analysis adapters."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from app.adapters.base import BaseFaceAnalyzer, FaceAnalysisResult


class BaseCCTVFaceAnalyzer(BaseFaceAnalyzer, ABC):
    """Abstract base class for face analysis adapters with CCTV partial face decomposition."""
    pass
