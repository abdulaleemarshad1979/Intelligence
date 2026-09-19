"""Base interfaces for Person Re-Identification adapters."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from app.adapters.base import BaseReIDModel, ReIDResult


class BasePersonReIDModel(BaseReIDModel, ABC):
    """Abstract base class for person Re-ID model adapters."""
    pass
