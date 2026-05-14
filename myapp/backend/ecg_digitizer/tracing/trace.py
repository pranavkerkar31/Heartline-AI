"""Waveform tracing with skeleton and subpixel centerline refinement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecg_digitizer.config import DigitizerConfig
from ecg_digitizer.utils.image import contiguous_runs


@dataclass(slots=True)
class TraceResult:
    """Digitized lead waveform and diagnostics."""

    signal_norm: np.ndarray
    signal_mv: np.ndarray
    baseline_y: float
    peak_px: float
    quality: dict[str, object]
    y_pixels: np.ndarray
    mask: np.ndarray


def _trace_mask(crop_bgr: np.ndarray, cfg: DigitizerConfig, label_skip_fraction: float) -> np.ndarray:
    """Extract near-white waveform ink and suppress colored grid lines."""
    import cv2

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(crop_bgr)
    maxc = np.maximum.reduce([b, g, r]).astype(np.int16)
    minc = np.minimum.reduce([b, g, r]).astype(np.int16)
    saturation = maxc - minc
    mask = ((gray >= cfg.trace_brightness_min) & (saturation <= cfg.trace_saturation_max))
    mask |= ((gray >= 115) & (saturation <= 30))
    h, w = mask.shape
    mask[:, : int(label_skip_fraction * w)] = False
    mask[:2, :] = False
    mask[-2:, :] = False
    mask[:, :2] = False
    mask[:, -2:] = False

    col_count = mask.sum(axis=0)
    col_top = np.full(w, h)
    col_bot = np.zeros(w)
    for x in np.flatnonzero(mask.any(axis=0)):
        ys = np.flatnonzero(mask[:, x])
        col_top[x], col_bot[x] = ys[0], ys[-1]
    col_span = np.where(mask.any(axis=0), col_bot - col_top + 1, 0)

    # FIX: distinguish separators from QRS peaks by continuity.
    # A separator is a SOLID vertical line — nearly every pixel in the column
    # is filled (col_count ≈ col_span).
    # A QRS peak is tall but has GAPS — ink only where the waveform passes,
    # so col_count << col_span.
    fill_ratio = np.where(col_span > 0, col_count / np.maximum(col_span, 1), 0.0)

    bad_cols = (
        (col_span > cfg.artifact_column_span_fraction * h) &
        (col_count > cfg.artifact_column_density_fraction * h) &
        (fill_ratio > 0.55)   # solid fill = separator; sparse fill = QRS spike
    )

    # Edge label zones — but only if the column is also narrow/solid
    edge = int(cfg.edge_label_zone_fraction * w)
    edge_dense = col_count > 0.18 * h
    edge_solid = fill_ratio > 0.45
    bad_cols[:edge] |= edge_dense[:edge] & edge_solid[:edge]
    bad_cols[max(0, w - edge):] |= edge_dense[max(0, w - edge):] & edge_solid[max(0, w - edge):]

    mask[:, bad_cols] = False

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)
    return mask


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """Skeletonize a binary trace mask if scikit-image is installed."""
    try:
        from skimage.morphology import skeletonize
    except Exception:
        return mask
    return skeletonize(mask).astype(bool)


def _subpixel_centerline(mask: np.ndarray, crop_bgr: np.ndarray) -> np.ndarray:
    """
    Estimate one subpixel y-value per x-column.
    For normal trace columns: intensity-weighted centroid around skeleton.
    For tall narrow QRS spike columns: use the extreme endpoint farthest
    from the estimated baseline so the peak tip is not lost.
    """
    import cv2

    h, w = mask.shape
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(float)
    weights = np.clip(gray - 80.0, 0.0, None) * mask
    skeleton = _skeletonize(mask)

    # Rough baseline estimate from the modal row of the mask
    row_density = mask.sum(axis=1).astype(float)
    smooth_density = np.convolve(row_density, np.ones(9) / 9, mode="same")
    baseline_guess = float(np.argmax(smooth_density)) if smooth_density.max() > 0 else h / 2.0

    # A column is QRS-spike-like if it is tall but sparse (not a solid separator)
    col_count = mask.sum(axis=0).astype(float)
    col_top = np.full(w, h, dtype=float)
    col_bot = np.zeros(w, dtype=float)
    for x in np.flatnonzero(mask.any(axis=0)):
        ys = np.flatnonzero(mask[:, x])
        col_top[x] = float(ys[0])
        col_bot[x] = float(ys[-1])
    col_span = col_bot - col_top
    fill_ratio = np.where(col_span > 0, col_count / (col_span + 1), 0.0)

    # Tall (>12% of crop height) AND sparse (fill<0.55) = QRS spike tip
    qrs_spike_cols = (col_span > 0.12 * h) & (fill_ratio < 0.55)

    y = np.full(w, np.nan, dtype=float)
    for x in range(w):
        ys = np.flatnonzero(skeleton[:, x])
        if len(ys) == 0:
            ys = np.flatnonzero(mask[:, x])
        if len(ys) == 0:
            continue

        if qrs_spike_cols[x]:
            # Pick the endpoint farthest from baseline — that's the true peak tip
            top_y = float(col_top[x])
            bot_y = float(col_bot[x])
            y[x] = top_y if abs(top_y - baseline_guess) >= abs(bot_y - baseline_guess) else bot_y
        else:
            # Normal trace: intensity-weighted centroid
            lo = max(0, int(ys.min()) - 2)
            hi = min(h, int(ys.max()) + 3)
            rows = np.arange(lo, hi)
            col_w = weights[lo:hi, x]
            if col_w.sum() <= 0:
                y[x] = float(np.mean(ys))
            else:
                y[x] = float(np.sum(rows * col_w) / np.sum(col_w))
    return y


def _fill_short_gaps(y: np.ndarray, max_gap: int) -> np.ndarray:
    """Interpolate only short internal gaps."""
    out = y.copy()
    bad = ~np.isfinite(out)
    for start, end in contiguous_runs(bad):
        left, right = start - 1, end + 1
        if end - start + 1 <= max_gap and left >= 0 and right < len(out) and np.isfinite(out[left]) and np.isfinite(out[right]):
            xs = np.arange(start, end + 1)
            out[xs] = np.interp(xs, [left, right], [out[left], out[right]])
    return out


def _baseline_from_y(y: np.ndarray, height: int) -> float:
    """Estimate baseline as the modal row of the traced centerline."""
    finite = y[np.isfinite(y)]
    if len(finite) == 0:
        return float(height / 2)
    hist, _ = np.histogram(np.clip(finite, 0, height), bins=np.arange(0, height + 2))
    smooth = np.convolve(hist.astype(float), np.ones(7) / 7, mode="same")
    return float(np.argmax(smooth))


def _quadratic_refine_peak(signal_px: np.ndarray) -> np.ndarray:
    """Refine local extrema by a quadratic 3-point fit."""
    y = signal_px.astype(float).copy()
    for i in range(1, len(y) - 1):
        if abs(y[i]) <= abs(y[i - 1]) or abs(y[i]) <= abs(y[i + 1]):
            continue
        denom = y[i - 1] - 2 * y[i] + y[i + 1]
        if abs(denom) < 1e-6:
            continue
        offset = 0.5 * (y[i - 1] - y[i + 1]) / denom
        if abs(offset) <= 0.5:
            y[i] = y[i] - 0.25 * (y[i - 1] - y[i + 1]) * offset
    return y


def digitize_lead(crop_bgr: np.ndarray, px_per_mm: float, lead_name: str, config: DigitizerConfig | None = None, manual_baseline_y: float | None = None) -> TraceResult:
    """Digitize a single lead crop into normalized and mV signals."""
    cfg = config or DigitizerConfig()
    h, w = crop_bgr.shape[:2]
    mask = _trace_mask(crop_bgr, cfg, cfg.label_skip_for_lead(lead_name))
    y = _subpixel_centerline(mask, crop_bgr)
    y = _fill_short_gaps(y, max_gap=max(3, int(round(px_per_mm * cfg.viterbi_max_gap_mm))))
    
    if manual_baseline_y is not None:
        baseline_y = manual_baseline_y
    else:
        baseline_y = _baseline_from_y(y, h)
    signal_px = np.zeros(w, dtype=float)
    finite = np.isfinite(y)
    signal_px[finite] = baseline_y - y[finite]
    signal_px = _quadratic_refine_peak(signal_px)
    peak_px = float(np.max(np.abs(signal_px))) if signal_px.size else 0.0
    signal_norm = (signal_px / peak_px).astype(np.float32) if peak_px > 0 else np.zeros(w, dtype=np.float32)
    signal_mv = (signal_px / px_per_mm / cfg.mm_per_mv).astype(np.float32)
    peak_mv = float(peak_px / px_per_mm / cfg.mm_per_mv) if px_per_mm > 0 else 0.0
    issues: list[str] = []
    lo, hi = cfg.amplitude_bounds_mv.get(lead_name, (0.05, 4.0))
    if peak_mv * 2 < lo:
        issues.append(f"amplitude too low ({peak_mv * 2:.3f} mV < {lo} mV)")
    elif peak_mv * 2 > hi:
        issues.append(f"amplitude too high ({peak_mv * 2:.3f} mV > {hi} mV)")
    quality = {
        "score": float(max(0.0, 1.0 - 0.25 * len(issues))),
        "issues": issues,
        "peak_mv": peak_mv,
        "diagnostics": {"finite_fraction": float(np.mean(finite)), "mask_pixels": int(mask.sum())},
    }
    return TraceResult(signal_norm, signal_mv, baseline_y, peak_px, quality, y, mask)
