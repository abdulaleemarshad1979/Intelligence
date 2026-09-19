"""Footage enhancement module implementing real-time low-latency stream pre-processing
and selective on-demand candidate super-resolution for CCTV intelligence.

Architecture:
- Step 1 (Always-on, instant, ~1ms):
    * CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space.
      Brightens dark shadows and night footage without blowing out highlights.
    * Fast Edge-Preserving Denoising:
      Eliminates high-ISO sensor static and night-vision grain while keeping sharp silhouette edges.
- Step 2 (On-demand, candidate-only):
    * High-fidelity face & body crop enhancement prior to biometric embedding generation.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


class StreamEnhancer:
    """Always-on lightweight image pre-processor for live CCTV streams."""

    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        enable_clahe: bool = True,
        enable_denoise: bool = True
    ):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.enable_clahe = enable_clahe
        self.enable_denoise = enable_denoise
        self._clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size)

    def apply_clahe_lab(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE exclusively on the Luminance (L) channel in LAB color space.

        Preserves chromaticity and skin-tone color histograms while lifting deep shadows
        and preventing blown-out highlights from streetlights and headlights.
        """
        if frame is None or frame.size == 0:
            return frame

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self._clahe.apply(l)
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def apply_fast_denoise(self, frame: np.ndarray) -> np.ndarray:
        """Apply fast edge-preserving bilateral filtering.

        Eliminates grainy night-vision sensor static while preserving distinct clothing
        boundaries and skeletal silhouette edges required for detection and tracking.
        """
        if frame is None or frame.size == 0:
            return frame

        # d=5, sigmaColor=35, sigmaSpace=35 runs in ~1ms while effectively cleaning sensor static
        return cv2.bilateralFilter(frame, d=5, sigmaColor=35, sigmaSpace=35)

    def enhance_stream_frame(self, frame: np.ndarray) -> np.ndarray:
        """Execute the two-step always-on stream enhancement pipeline."""
        if frame is None or frame.size == 0:
            return frame

        enhanced = frame
        if self.enable_clahe:
            enhanced = self.apply_clahe_lab(enhanced)
        if self.enable_denoise:
            enhanced = self.apply_fast_denoise(enhanced)
        return enhanced

    def get_status(self) -> Dict[str, Any]:
        return {
            "clahe_enabled": self.enable_clahe,
            "denoise_enabled": self.enable_denoise,
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_grid_size": list(self.clahe_grid_size),
            "step": "STEP_1_ALWAYS_ON"
        }


class CandidateEnhancer:
    """Selective, on-demand AI and edge-guided crop enhancer.

    Rule of thumb: Only triggered on candidate person/face crops right before gallery matching,
    preventing server throughput degradation on high-framerate multi-camera live streams.
    """

    def __init__(self, target_face_size: Tuple[int, int] = (160, 160)):
        self.target_face_size = target_face_size
        self._gfpgan_available = False
        self._esrgan_available = False

    def enhance_face_crop(self, face_crop: np.ndarray) -> np.ndarray:
        """Reconstruct and sharpen a facial crop before running feature extraction (YuNet/SFace).

        Applies unsharp mask sharpening and high-order Lanczos interpolation to reconstruct
        facial geometry when neural weights (GFPGAN) are unavailable.
        """
        if face_crop is None or face_crop.size == 0:
            return face_crop

        h, w = face_crop.shape[:2]
        # Upscale small face crops with Lanczos4 interpolation
        if h < self.target_face_size[1] or w < self.target_face_size[0]:
            upscaled = cv2.resize(face_crop, self.target_face_size, interpolation=cv2.INTER_LANCZOS4)
        else:
            upscaled = face_crop.copy()

        # Unsharp masking for crisp facial feature restoration (eyes, nose bridge, lips)
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.5)
        sharpened = cv2.addWeighted(upscaled, 1.4, gaussian, -0.4, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def enhance_body_crop(self, person_crop: np.ndarray, scale_factor: float = 1.5) -> np.ndarray:
        """Enhance person body crop resolution and contrast prior to OSNet ReID embedding."""
        if person_crop is None or person_crop.size == 0:
            return person_crop

        h, w = person_crop.shape[:2]
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        upscaled = cv2.resize(person_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Mild edge sharpening for cloth textures and silhouettes
        kernel = np.array([
            [0, -0.5, 0],
            [-0.5, 3.0, -0.5],
            [0, -0.5, 0]
        ], dtype=np.float32)
        sharpened = cv2.filter2D(upscaled, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)


# Default singleton instances
stream_enhancer = StreamEnhancer()
candidate_enhancer = CandidateEnhancer()
