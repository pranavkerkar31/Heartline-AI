"""Robust ECG grid calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecg_digitizer.config import DigitizerConfig


try:
    from scipy.signal import find_peaks as _find_peaks
except Exception:  # pragma: no cover - fallback used when scipy is unavailable
    _find_peaks = None


@dataclass(slots=True)
class CalibrationResult:
    """Pixel-per-mm estimate with diagnostics."""

    px_per_mm: float
    confidence: float
    method: str
    candidates: list[float]
    details: dict[str, float | str]


def _peaks_1d(values: np.ndarray, min_distance: int, min_height: float, prominence: float) -> np.ndarray:
    """Small wrapper around scipy peaks with a deterministic NumPy fallback."""
    if _find_peaks is not None:
        peaks, _ = _find_peaks(values, height=min_height, distance=min_distance, prominence=prominence)
        return peaks.astype(int)
    peaks: list[int] = []
    last = -min_distance
    for i in range(1, len(values) - 1):
        if i - last < min_distance:
            continue
        if values[i] >= min_height and values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            left = max(0, i - min_distance)
            right = min(len(values), i + min_distance + 1)
            if values[i] - np.median(values[left:right]) >= prominence:
                peaks.append(i)
                last = i
    return np.asarray(peaks, dtype=int)


def _colored_grid_mask(img: np.ndarray) -> np.ndarray:
    """Detect saturated red/green ECG grid pixels."""
    b, g, r = [c.astype(np.float32) for c in np.moveaxis(img, 2, 0)]
    red = (r > 70) & (r > 1.30 * g) & (r > 1.30 * b)
    green = (g > 45) & (g > 1.20 * r) & (g > 1.20 * b)
    return red | green


def _spacing_from_projection(mask: np.ndarray, axis: int, config: DigitizerConfig) -> tuple[float | None, int]:
    """Estimate grid spacing from row/column projection peaks."""
    density = mask.sum(axis=axis).astype(float)
    if density.size < 16 or density.max() <= 0:
        return None, 0
    smooth = np.convolve(density, np.ones(3) / 3, mode="same")
    peaks = _peaks_1d(
        smooth,
        min_distance=5,
        min_height=max(3.0, 0.15 * float(smooth.max())),
        prominence=max(1.0, 0.05 * float(smooth.max())),
    )
    diffs = np.diff(peaks)
    diffs = diffs[
        (diffs >= config.calibration_min_px_per_mm) &
        (diffs <= config.calibration_max_px_per_mm)
    ]
    if len(diffs) < 4:
        return None, len(diffs)
    return float(np.median(diffs)), len(diffs)


def _fft_spacing(gray: np.ndarray, config: DigitizerConfig) -> float | None:
    """Estimate small-square grid spacing from the dominant FFT frequency."""
    h, w = gray.shape
    strip = gray[h // 4: 3 * h // 4, w // 8: 7 * w // 8].astype(np.float32)
    if strip.size == 0:
        return None
    col_mean = strip.mean(axis=0)
    col_mean -= col_mean.mean()
    fft = np.abs(np.fft.rfft(col_mean))
    freqs = np.fft.rfftfreq(len(col_mean))
    if len(fft) < 8:
        return None
    fft[0] = 0
    fft[: max(1, int(len(fft) * 0.02))] = 0
    freq = freqs[int(np.argmax(fft))]
    if freq <= 0:
        return None
    spacing = float(1.0 / freq)
    if config.calibration_min_px_per_mm <= spacing <= config.calibration_max_px_per_mm:
        return spacing
    return None


def estimate_px_per_mm(img: np.ndarray, config: DigitizerConfig | None = None) -> CalibrationResult:
    """Estimate ECG small-grid spacing using multi-region color and FFT consensus."""
    import cv2

    cfg = config or DigitizerConfig()
    h, w = img.shape[:2]
    candidates: list[float] = []
    counts: list[int] = []

    for x0f, y0f, x1f, y1f in cfg.calibration_regions:
        x0, y0 = int(x0f * w), int(y0f * h)
        x1, y1 = int(x1f * w), int(y1f * h)
        roi = img[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        mask = _colored_grid_mask(roi)
        for axis in (0, 1):
            spacing, count = _spacing_from_projection(mask, axis=axis, config=cfg)
            if spacing is not None:
                candidates.append(spacing)
                counts.append(count)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fft_spacing = _fft_spacing(gray, cfg)
    if fft_spacing is not None:
        candidates.append(fft_spacing)
        counts.append(3)

    if not candidates:
        return CalibrationResult(
            px_per_mm=cfg.default_px_per_mm,
            confidence=0.0,
            method="fallback-default",
            candidates=[],
            details={"reason": "no valid grid candidates"},
        )

    arr = np.asarray(candidates, dtype=float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    inliers = arr[np.abs(arr - median) <= max(1.5, 2.5 * mad)]
    if len(inliers) == 0:
        inliers = arr
    px_per_mm = float(np.median(inliers))
    spread = float(np.std(inliers)) if len(inliers) > 1 else 0.0
    support = min(1.0, len(inliers) / 6.0)
    confidence = float(np.clip(support * (1.0 / (1.0 + spread)), 0.0, 1.0))
    return CalibrationResult(
        px_per_mm=px_per_mm,
        confidence=confidence,
        method="multi-region-consensus",
        candidates=[float(x) for x in arr],
        details={"n_candidates": float(len(arr)), "spread": spread},
    )
