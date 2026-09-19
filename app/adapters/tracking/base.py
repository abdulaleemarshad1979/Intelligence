"""Base interfaces for Multi-Object Tracking adapters."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from app.adapters.base import BaseTracker, DetectionResult, TrackingResult


class BaseMOTTracker(BaseTracker, ABC):
    """Abstract base class for multi-object tracking adapters."""
    pass
