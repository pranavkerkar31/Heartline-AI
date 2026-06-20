"""Signal-space validation metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _zscore(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=float)
    std = float(np.nanstd(y))
    if std < 1e-12:
        return np.zeros_like(y)
    return (y - float(np.nanmean(y))) / std


def _align_by_cross_correlation(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Align two same-rate signals by cross-correlation lag."""
    a = _zscore(reference)
    b = _zscore(estimate)
    corr = np.correlate(a, b, mode="full")
    lag = int(np.argmax(corr) - (len(b) - 1))
    if lag > 0:
        ref = reference[lag:]
        est = estimate[:len(ref)]
    elif lag < 0:
        est = estimate[-lag:]
        ref = reference[:len(est)]
    else:
        ref, est = reference, estimate
    n = min(len(ref), len(est))
    return ref[:n], est[:n], lag


def _dtw_distance(a: np.ndarray, b: np.ndarray, window: int | None = None) -> float:
    """Compute DTW distance with optional Sakoe-Chiba window."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan")
    w = max(abs(n - m), window or max(n, m))
    prev = np.full(m + 1, np.inf)
    curr = np.full(m + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr[:] = np.inf
        lo = max(1, i - w)
        hi = min(m, i + w)
        for j in range(lo, hi + 1):
            cost = abs(a[i - 1] - b[j - 1])
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return float(prev[m] / (n + m))


def _simple_peaks(signal: np.ndarray, sample_hz: float) -> np.ndarray:
    """Detect R-peak candidates using a simple refractory local-maximum rule."""
    y = _zscore(np.asarray(signal, dtype=float))
    refractory = max(1, int(0.22 * sample_hz))
    threshold = max(0.7, float(np.nanpercentile(y, 90)))
    peaks: list[int] = []
    last = -refractory
    for i in range(1, len(y) - 1):
        if i - last < refractory:
            continue
        if y[i] >= threshold and y[i] >= y[i - 1] and y[i] >= y[i + 1]:
            peaks.append(i)
            last = i
    return np.asarray(peaks, dtype=int)


def _clinical_metrics(reference: np.ndarray, estimate: np.ndarray, sample_hz: float) -> dict[str, float]:
    """Estimate simple interval/amplitude errors from detected R peaks."""
    r_ref = _simple_peaks(reference, sample_hz)
    r_est = _simple_peaks(estimate, sample_hz)
    out = {
        "rr_error_ms": float("nan"),
        "r_peak_amplitude_error_mv": float("nan"),
        "qrs_duration_error_ms": float("nan"),
        "qt_error_ms": float("nan"),
    }
    if len(r_ref) >= 2 and len(r_est) >= 2:
        rr_ref = np.diff(r_ref) / sample_hz * 1000.0
        rr_est = np.diff(r_est) / sample_hz * 1000.0
        n = min(len(rr_ref), len(rr_est))
        out["rr_error_ms"] = float(np.mean(np.abs(rr_ref[:n] - rr_est[:n])))
    if len(r_ref) and len(r_est):
        n = min(len(r_ref), len(r_est))
        out["r_peak_amplitude_error_mv"] = float(np.mean(np.abs(reference[r_ref[:n]] - estimate[r_est[:n]])))
    # QRS/QT require delineation. Provide deterministic placeholders based on R alignment,
    # so the report schema remains stable until wave boundary delineation is added.
    return out


def validate_pair(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_hz: float,
    zscore: bool = False,
    physical_mv: bool = True,
) -> dict[str, float]:
    """Validate one estimated signal against a reference signal."""
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    ref, est, lag = _align_by_cross_correlation(ref, est)
    metrics_ref = _zscore(ref) if zscore else ref
    metrics_est = _zscore(est) if zscore else est
    diff = metrics_est - metrics_ref
    rmse = float(np.sqrt(np.mean(diff ** 2))) if len(diff) else float("nan")
    mae = float(np.mean(np.abs(diff))) if len(diff) else float("nan")
    denom = float(np.sqrt(np.sum(metrics_ref ** 2)))
    prd = float(100.0 * np.sqrt(np.sum(diff ** 2)) / denom) if denom > 1e-12 else float("nan")
    noise = float(np.sum(diff ** 2))
    signal = float(np.sum(metrics_ref ** 2))
    snr = float(10.0 * np.log10(signal / noise)) if noise > 1e-12 else float("inf")
    pearson = float(np.corrcoef(metrics_ref, metrics_est)[0, 1]) if len(metrics_ref) > 1 else float("nan")
    out = {
        "pearson": pearson,
        "rmse_mv" if physical_mv else "rmse": rmse,
        "mae_mv" if physical_mv else "mae": mae,
        "prd": prd,
        "snr": snr,
        "dtw": _dtw_distance(metrics_ref, metrics_est, window=max(10, int(0.1 * len(metrics_ref)))),
        "lag_samples": float(lag),
        "lag_ms": float(1000.0 * lag / sample_hz),
    }
    out.update(_clinical_metrics(ref, est, sample_hz))
    return out


def validate_signals(
    reference_by_lead: dict[str, np.ndarray],
    estimate_by_lead: dict[str, np.ndarray],
    sample_hz: float,
    out_csv: str | Path | None = None,
    out_json: str | Path | None = None,
) -> dict[str, dict[str, float]]:
    """Validate all common leads and optionally export CSV/JSON reports."""
    report = {
        lead: validate_pair(reference_by_lead[lead], estimate_by_lead[lead], sample_hz)
        for lead in reference_by_lead.keys() & estimate_by_lead.keys()
    }
    if out_json is not None:
        Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if out_csv is not None:
        import csv
        keys = sorted({k for metrics in report.values() for k in metrics})
        with Path(out_csv).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["lead", *keys])
            writer.writeheader()
            for lead, metrics in report.items():
                writer.writerow({"lead": lead, **metrics})
    return report
