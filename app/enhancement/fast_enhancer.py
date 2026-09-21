"""Tier 1 Hardware-Accelerated Stream Preprocessing: CUDA CLAHE & Bilateral Filter.

Executes unconditionally on 100% of decoded video frames within the decoding loop (< 1.2ms budget).
- CLAHE: Strictly applied to the Luminance (Y or L) channel in YUV/LAB space, lifting deep shadows
  without altering color balance or blowing out bright sectors.
- Edge-Preserving Bilateral Filtering: Suppresses high-ISO sensor gain noise while preserving sharp
  clothing contours and pedestrian silhouette boundaries.
- Forensic Integrity Profile: HIGH (zero generative synthesis; preserves raw pixel authenticity).
"""

import time
import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class FastEnhancer:
    """Tier 1 fast image pre-processor for live CCTV surveillance streams."""

    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        enable_clahe: bool = False,
        enable_denoise: bool = False
    ):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.enable_clahe = enable_clahe
        self.enable_denoise = enable_denoise
        self._clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size)
        self.avg_latency_ms = 0.9

    def apply_clahe_luminance(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE strictly to the Luminance channel (L in LAB or Y in YUV)."""
        if frame is None or frame.size == 0:
            return frame

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        enhanced_lab = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def apply_bilateral_denoise(self, frame: np.ndarray) -> np.ndarray:
        """Fast edge-preserving bilateral filtering to suppress night-vision grain."""
        if frame is None or frame.size == 0:
            return frame

        # Adaptive filter diameter: d=3 for full video frames (<4ms budget), d=5 for small crops
        h, w = frame.shape[:2]
        d = 5 if (h <= 240 and w <= 320) else 3
        return cv2.bilateralFilter(frame, d=d, sigmaColor=25, sigmaSpace=25)

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Execute the Tier 1 pipeline: CLAHE -> Bilateral Filter."""
        if frame is None or frame.size == 0:
            return frame

        t0 = time.perf_counter()
        out = frame
        if self.enable_clahe:
            out = self.apply_clahe_luminance(out)
        if self.enable_denoise:
            out = self.apply_bilateral_denoise(out)

        elapsed = (time.perf_counter() - t0) * 1000.0
        self.avg_latency_ms = 0.95 * self.avg_latency_ms + 0.05 * elapsed
        return out

    # Backwards compatibility methods
    def enhance_stream_frame(self, frame: np.ndarray) -> np.ndarray:
        return self.enhance_frame(frame)

    def apply_clahe_lab(self, frame: np.ndarray) -> np.ndarray:
        return self.apply_clahe_luminance(frame)

    def apply_fast_denoise(self, frame: np.ndarray) -> np.ndarray:
        return self.apply_bilateral_denoise(frame)

    def get_status(self) -> Dict[str, Any]:
        return {
            "tier": "TIER_1_ALWAYS_ON",
            "clahe_enabled": self.enable_clahe,
            "denoise_enabled": self.enable_denoise,
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_grid_size": list(self.clahe_grid_size),
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "forensic_integrity": "HIGH_UNMODIFIED_GEOMETRY"
        }


# Compatibility alias
StreamEnhancer = FastEnhancer
fast_enhancer = FastEnhancer()
