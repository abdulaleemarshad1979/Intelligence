"""Re-export of Person Re-ID adapters for backward compatibility."""

from app.adapters.reid.osnet import OSNetReIDAdapter
from app.adapters.reid.fastreid import FastReIDAdapter

__all__ = ["OSNetReIDAdapter", "FastReIDAdapter"]
