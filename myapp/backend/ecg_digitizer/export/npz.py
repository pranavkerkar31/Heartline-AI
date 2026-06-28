"""NPZ export preserving physical ECG time and voltage units."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ecg_digitizer.config import DigitizerConfig
from ecg_digitizer.segmentation import LeadSegment
from ecg_digitizer.utils.image import ensure_dir


def resample_to_500hz(
    signal: np.ndarray,
    config: DigitizerConfig,
    pixels_per_mm_x: float | None,
) -> np.ndarray:
    """Convert one horizontal pixel sample per column to a physical 500 Hz trace."""

    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return np.zeros(0, dtype=np.float32)
    if not pixels_per_mm_x or pixels_per_mm_x <= 0:
        raise ValueError("A valid horizontal pixels/mm calibration is required")

    seconds_per_pixel = 1.0 / (pixels_per_mm_x * config.mm_per_sec)
    source_times = np.arange(signal.size, dtype=float) * seconds_per_pixel
    duration = float(source_times[-1])
    if duration <= 0:
        return signal[:1].astype(np.float32)

    target_dt = 1.0 / config.sample_hz
    target_count = int(np.floor(duration / target_dt)) + 1
    target_times = np.arange(target_count, dtype=float) * target_dt
    return np.interp(target_times, source_times, signal).astype(np.float32)


def save_signals_npz(
    signals_norm: dict[str, np.ndarray],
    signals_mv: dict[str, np.ndarray],
    metadata: dict[str, tuple[float, float]],
    outdir: str | Path,
    segments: dict[str, LeadSegment] | None = None,
    config: DigitizerConfig | None = None,
    pixels_per_mm_x: float | None = None,
    pixels_per_mm_y: float | None = None,
) -> tuple[Path, Path]:
    """Save normalized and mV NPZ files after physical 500 Hz interpolation."""

    cfg = config or DigitizerConfig()
    out = ensure_dir(outdir)

    norm_dict: dict[str, np.ndarray | np.float32] = {
        name: resample_to_500hz(sig, cfg, pixels_per_mm_x)
        for name, sig in signals_norm.items()
    }
    for name, (baseline, peak) in metadata.items():
        norm_dict[f"{name}__baseline"] = np.float32(baseline)
        norm_dict[f"{name}__peak"] = np.float32(peak)
    if segments:
        for name, segment in segments.items():
            norm_dict[f"{name}__box"] = np.asarray(segment.box, dtype=np.int32)

    norm_dict["pixels_per_mm_x"] = np.float32(pixels_per_mm_x)
    norm_dict["pixels_per_mm_y"] = np.float32(pixels_per_mm_y)
    path_norm = out / "ecg_signals.npz"
    np.savez(path_norm, **norm_dict)

    mv_dict: dict[str, np.ndarray | np.float64] = {
        (name if name != "Rhythm" else "RHYTHM"): resample_to_500hz(
            sig,
            cfg,
            pixels_per_mm_x,
        )
        for name, sig in signals_mv.items()
    }
    mv_dict["sampling_rate"] = np.float64(cfg.sample_hz)
    mv_dict["pixels_per_mm_x"] = np.float64(pixels_per_mm_x)
    mv_dict["pixels_per_mm_y"] = np.float64(pixels_per_mm_y)
    mv_dict["mm_per_second"] = np.float64(cfg.mm_per_sec)
    mv_dict["mm_per_mv"] = np.float64(cfg.mm_per_mv)
    path_mv = out / "ecg_signals_mv.npz"
    np.savez(path_mv, **mv_dict)
    return path_norm, path_mv
