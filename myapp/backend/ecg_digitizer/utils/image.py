"""Image loading and low-level mask helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_image(path: str | Path):
    """Load an image with OpenCV in BGR order."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def is_dark_background(gray: np.ndarray) -> bool:
    """Return True when the image background is predominantly dark."""
    return float(np.mean(gray)) < 128.0


def foreground_mask(gray: np.ndarray) -> np.ndarray:
    """Compute an Otsu foreground mask with foreground=True."""
    import cv2

    if is_dark_background(gray):
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return gray > threshold
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return gray < threshold


def contiguous_runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive contiguous True-index runs."""
    xs = np.flatnonzero(mask_1d)
    if len(xs) == 0:
        return []
    breaks = np.where(np.diff(xs) > 1)[0]
    starts = np.r_[xs[0], xs[breaks + 1]]
    ends = np.r_[xs[breaks], xs[-1]]
    return list(zip(starts.astype(int), ends.astype(int)))
