"""Re-ID adapters package."""

from app.adapters.reid.base import BasePersonReIDModel
from app.adapters.reid.osnet import OSNetReIDAdapter
from app.adapters.reid.fastreid import FastReIDAdapter

__all__ = ["BasePersonReIDModel", "OSNetReIDAdapter", "FastReIDAdapter"]
