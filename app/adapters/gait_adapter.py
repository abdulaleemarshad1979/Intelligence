"""Re-export of Gait Dynamics adapters for backward compatibility."""

from app.adapters.gait.gaitset import GaitSetAdapter
from app.adapters.gait.opengait import OpenGaitAdapter

GaitSetDynamicsAdapter = GaitSetAdapter

__all__ = ["GaitSetAdapter", "OpenGaitAdapter", "GaitSetDynamicsAdapter"]
