"""NPZ export preserving the notebook schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ecg_digitizer.config import DigitizerConfig
from ecg_digitizer.segmentation import LeadSegment
from ecg_digitizer.utils.image import ensure_dir


def resample_to_mendeley(signal: np.ndarray, lead_name: str, config: DigitizerConfig) -> np.ndarray:
    """Resample a lead to the notebook/Mendeley-compatible sample count."""
    target_n = config.target_samples_for_lead(lead_name)
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return np.zeros(target_n, dtype=np.float32)
    if signal.size == target_n:
        return signal.astype(np.float32)
    src_x = np.linspace(0.0, 1.0, signal.size)
    dst_x = np.linspace(0.0, 1.0, target_n)
    return np.interp(dst_x, src_x, signal).astype(np.float32)


def save_signals_npz(
    signals_norm: dict[str, np.ndarray],
    signals_mv: dict[str, np.ndarray],
    metadata: dict[str, tuple[float, float]],
    outdir: str | Path,
    segments: dict[str, LeadSegment] | None = None,
    config: DigitizerConfig | None = None,
) -> tuple[Path, Path]:
    """Save normalized and mV NPZ files with the existing notebook schema."""
    cfg = config or DigitizerConfig()
    out = ensure_dir(outdir)

    norm_dict: dict[str, np.ndarray | np.float32] = {
        name: resample_to_mendeley(sig, name, cfg) for name, sig in signals_norm.items()
    }
    for name, (baseline, peak) in metadata.items():
        norm_dict[f"{name}__baseline"] = np.float32(baseline)
        norm_dict[f"{name}__peak"] = np.float32(peak)
    if segments:
        for name, segment in segments.items():
            norm_dict[f"{name}__box"] = np.asarray(segment.box, dtype=np.int32)
    path_norm = out / "ecg_signals.npz"
    np.savez(path_norm, **norm_dict)

    mv_dict: dict[str, np.ndarray | np.float64] = {
        (name if name != "Rhythm" else "RHYTHM"): resample_to_mendeley(sig, name, cfg)
        for name, sig in signals_mv.items()
    }
    mv_dict["sampling_rate"] = np.float64(cfg.sample_hz)
    path_mv = out / "ecg_signals_mv.npz"
    np.savez(path_mv, **mv_dict)
    return path_norm, path_mv
