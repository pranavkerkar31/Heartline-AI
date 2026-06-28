"""Automatic ECG grid calibration using row/column autocorrelation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class AxisCalibration:
    pixels_per_mm: float
    major_period_px: float
    confidence: float
    candidates: list[tuple[float, float]]


@dataclass(slots=True)
class GridCalibration:
    pixels_per_mm_x: float
    pixels_per_mm_y: float
    confidence: float
    x_axis: AxisCalibration
    y_axis: AxisCalibration
    paper_tone: int


def _autocorrelation_candidates(
    projection: np.ndarray,
    search_min: int,
    search_max: int,
) -> list[tuple[float, float]]:
    from scipy.signal import find_peaks

    centered = projection.astype(float) - float(np.mean(projection))
    if centered.size < 2 or not np.any(np.abs(centered) > 1e-9):
        return []

    fft = np.fft.rfft(centered, n=2 * centered.size)
    acf = np.fft.irfft(fft * np.conj(fft))[: centered.size].real
    acf /= float(acf[0]) + 1e-9

    hi = min(search_max, acf.size - 2)
    lo = min(max(2, search_min), hi)
    peaks, properties = find_peaks(acf[lo : hi + 1], height=0.03)
    candidates = [
        (float(index + lo), float(height))
        for index, height in zip(peaks, properties.get("peak_heights", []))
    ]
    return sorted(candidates, key=lambda item: item[1], reverse=True)


def _select_major_period(
    candidates: list[tuple[float, float]],
    expected_major_px: float,
    reference_major_px: float | None = None,
) -> AxisCalibration:
    target = reference_major_px or expected_major_px
    plausible = [
        item
        for item in candidates
        if 0.55 * target <= item[0] <= 1.55 * target
    ]

    if plausible:
        period, strength = max(
            plausible,
            key=lambda item: item[1] * np.exp(-1.8 * abs(np.log(item[0] / target))),
        )
    elif candidates:
        period, strength = candidates[0]
        # A strong minor-grid harmonic may be returned instead of the 5 mm period.
        if period < 0.45 * expected_major_px:
            period *= 5.0
    else:
        period, strength = target, 0.0

    pixels_per_mm = float(period / 5.0)
    proximity = float(np.exp(-abs(np.log(max(period, 1e-6) / max(target, 1e-6)))))
    confidence = float(np.clip(strength * 3.0, 0.0, 1.0) * proximity)
    return AxisCalibration(
        pixels_per_mm=pixels_per_mm,
        major_period_px=float(period),
        confidence=confidence,
        candidates=candidates[:8],
    )


def detect_grid_calibration(image: np.ndarray) -> GridCalibration:
    """Measure pixels/mm independently along X and Y from 5 mm grid boxes."""

    import cv2

    if image is None or image.size == 0:
        raise ValueError("Cannot calibrate an empty ECG image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    height, width = gray.shape

    histogram = np.bincount(gray.ravel(), minlength=256)
    paper_tone = int(np.argmax(histogram[180:]) + 180)
    grid_hi = max(0, paper_tone - 2)
    grid_lo = max(0, paper_tone - 40)
    grid_mask = ((gray >= grid_lo) & (gray <= grid_hi)).astype(np.float32)

    # A standard four-column ECG spans about 250 mm horizontally.
    estimated_px_per_mm = max(1.0, width / 250.0)
    expected_major = 5.0 * estimated_px_per_mm
    search_min = max(5, int(0.45 * expected_major))
    search_max = min(250, int(1.8 * expected_major))

    x_candidates = _autocorrelation_candidates(
        grid_mask.sum(axis=0),
        search_min,
        search_max,
    )
    x_axis = _select_major_period(x_candidates, expected_major)

    y_candidates = _autocorrelation_candidates(
        grid_mask.sum(axis=1),
        search_min,
        search_max,
    )
    y_axis = _select_major_period(
        y_candidates,
        expected_major,
        reference_major_px=x_axis.major_period_px,
    )

    ratio = y_axis.pixels_per_mm / max(x_axis.pixels_per_mm, 1e-9)
    square_consistency = float(np.exp(-3.0 * abs(np.log(max(ratio, 1e-9)))))
    confidence = float(min(x_axis.confidence, y_axis.confidence) * square_consistency)

    return GridCalibration(
        pixels_per_mm_x=x_axis.pixels_per_mm,
        pixels_per_mm_y=y_axis.pixels_per_mm,
        confidence=confidence,
        x_axis=x_axis,
        y_axis=y_axis,
        paper_tone=paper_tone,
    )
