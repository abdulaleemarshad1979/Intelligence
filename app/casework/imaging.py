"""Still-image tools: crop, enhancement operations and side-by-side comparison.

Each function reads a path and writes a new file; none modifies its input. Callers register
the result through CaseService.add_derived so it is labelled and logged.
"""

from typing import Any, Callable, Dict, Tuple

import cv2
import numpy as np


def _read(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unreadable image: {path}")
    return img


def _write(img: np.ndarray, path: str) -> None:
    if not cv2.imwrite(path, img):
        raise ValueError(f"Could not write {path}")


def crop(src: str, dst: str, x: int, y: int, w: int, h: int) -> Dict[str, Any]:
    img = _read(src)
    H, W = img.shape[:2]
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > W or y + h > H:
        raise ValueError(f"Crop box {(x, y, w, h)} falls outside the {W}x{H} image")
    _write(img[y:y + h, x:x + w], dst)
    return {"x": x, "y": y, "w": w, "h": h, "source_size": [W, H]}


def _brightness_contrast(img, alpha: float = 1.0, beta: float = 0.0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def _gamma(img, gamma: float = 1.0):
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def _clahe(img, clip_limit: float = 2.0, tile: int = 8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _unsharp(img, radius: float = 1.5, amount: float = 1.0):
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def _denoise(img, strength: float = 5.0):
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def _upscale(img, factor: float = 2.0):
    if not 1.0 < factor <= 8.0:
        raise ValueError("factor must be in (1, 8]")
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


OPERATIONS: Dict[str, Callable[..., np.ndarray]] = {
    "BRIGHTNESS_CONTRAST": _brightness_contrast,
    "GAMMA": _gamma,
    "CLAHE": _clahe,
    "UNSHARP_MASK": _unsharp,
    "DENOISE": _denoise,
    "UPSCALE_BICUBIC": _upscale,
}


def enhance(src: str, dst: str, operation: str, **params: Any) -> Dict[str, Any]:
    """Classical, deterministic adjustments only. No generative or learned face restoration:
    those invent detail and must not be used on evidence."""
    if operation not in OPERATIONS:
        raise ValueError(f"Unknown operation {operation!r}; choose from {sorted(OPERATIONS)}")
    _write(OPERATIONS[operation](_read(src), **params), dst)
    return dict(params)


def _fit_height(img: np.ndarray, height: int) -> np.ndarray:
    scale = height / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)


def _caption(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((28, img.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(bar, text[:80], (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def side_by_side(src_a: str, src_b: str, dst: str, label_a: str, label_b: str,
                 height: int = 640) -> Dict[str, Any]:
    """Two images at the same display height, captioned, for manual comparison."""
    a, b = _fit_height(_read(src_a), height), _fit_height(_read(src_b), height)
    gap = np.full((height + 28, 8, 3), 255, dtype=np.uint8)
    _write(np.hstack([_caption(a, label_a), gap, _caption(b, label_b)]), dst)
    return {"display_height": height, "label_a": label_a, "label_b": label_b}
