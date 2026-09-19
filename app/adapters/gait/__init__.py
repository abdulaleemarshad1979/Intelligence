"""Gait recognition adapters package."""

from app.adapters.gait.base import GaitAdapter
from app.adapters.gait.opengait import OpenGaitAdapter
from app.adapters.gait.gaitset import GaitSetAdapter

__all__ = ["GaitAdapter", "OpenGaitAdapter", "GaitSetAdapter"]
