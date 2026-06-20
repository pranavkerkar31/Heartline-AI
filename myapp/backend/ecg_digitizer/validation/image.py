"""Image-space reprojection validation metrics."""

from __future__ import annotations

import numpy as np


def _directed_hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    max_min = 0.0
    for chunk_start in range(0, len(a), 512):
        chunk = a[chunk_start:chunk_start + 512]
        d2 = np.sum((chunk[:, None, :] - b[None, :, :]) ** 2, axis=2)
        max_min = max(max_min, float(np.sqrt(np.min(d2, axis=1)).max()))
    return max_min


def reprojection_metrics(trace_mask: np.ndarray, projected_mask: np.ndarray) -> dict[str, float]:
    """Compare projected digitized waveform pixels against original trace pixels."""
    trace = np.asarray(trace_mask, dtype=bool)
    proj = np.asarray(projected_mask, dtype=bool)
    if trace.shape != proj.shape:
        raise ValueError(f"Mask shapes differ: {trace.shape} != {proj.shape}")
    inter = np.logical_and(trace, proj).sum()
    union = np.logical_or(trace, proj).sum()
    dice = float(2 * inter / max(1, trace.sum() + proj.sum()))
    iou = float(inter / max(1, union))
    coverage = float(inter / max(1, trace.sum()))
    trace_pts = np.argwhere(trace)
    proj_pts = np.argwhere(proj)
    if len(trace_pts) and len(proj_pts):
        # Distance from each projected point to nearest original waveform point.
        min_d: list[np.ndarray] = []
        for start in range(0, len(proj_pts), 512):
            chunk = proj_pts[start:start + 512]
            d2 = np.sum((chunk[:, None, :] - trace_pts[None, :, :]) ** 2, axis=2)
            min_d.append(np.sqrt(np.min(d2, axis=1)))
        distances = np.concatenate(min_d)
        pixel_rmse = float(np.sqrt(np.mean(distances ** 2)))
        mean_distance = float(np.mean(distances))
        max_deviation = float(np.max(distances))
        hausdorff = max(_directed_hausdorff(trace_pts, proj_pts), _directed_hausdorff(proj_pts, trace_pts))
    else:
        pixel_rmse = mean_distance = max_deviation = hausdorff = float("nan")
    return {
        "pixel_rmse": pixel_rmse,
        "mean_pixel_distance": mean_distance,
        "max_deviation": max_deviation,
        "hausdorff_distance": hausdorff,
        "waveform_coverage_pct": 100.0 * coverage,
        "dice": dice,
        "iou": iou,
    }
