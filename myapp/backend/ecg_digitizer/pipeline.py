"""End-to-end digitization pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ecg_digitizer.calibration import CalibrationResult, estimate_px_per_mm
from ecg_digitizer.config import DigitizerConfig
from ecg_digitizer.export import save_signals_npz
from ecg_digitizer.segmentation import LeadSegment, SegmentReport, segment_leads
from ecg_digitizer.tracing import TraceResult, digitize_lead
from ecg_digitizer.utils.image import ensure_dir, load_image
from ecg_digitizer.visualization import save_npz_overlay


@dataclass(slots=True)
class DigitizationResult:
    """Outputs from one ECG digitization run."""

    calibration: CalibrationResult
    segments: dict[str, LeadSegment]
    traces: dict[str, TraceResult]
    segment_report: SegmentReport
    npz_norm_path: Path
    npz_mv_path: Path
    overlay_path: Path


def run_digitization(
    input_path: str | Path,
    output_dir: str | Path = "outputs/ecg_may",
    config: DigitizerConfig | None = None,
) -> DigitizationResult:
    """Run calibration, segmentation, tracing, NPZ export, and overlay generation."""
    cfg = config or DigitizerConfig(output_dir=Path(output_dir))
    out = ensure_dir(output_dir)
    img = load_image(input_path)
    calibration = estimate_px_per_mm(img, cfg)
    if cfg.manual_px_per_mm is not None:
        calibration.px_per_mm = cfg.manual_px_per_mm
        calibration.confidence = 1.0

    segments, segment_report = segment_leads(img, out, cfg)

    traces: dict[str, TraceResult] = {}
    signals_norm: dict[str, np.ndarray] = {}
    signals_mv: dict[str, np.ndarray] = {}
    metadata: dict[str, tuple[float, float]] = {}
    for lead in cfg.lead_order:
        segment = segments[lead]
        # Determine the row index for standard 12-lead + rhythm layout
        if lead in ("I", "aVR", "V1", "V4"): row_idx = 0
        elif lead in ("II", "aVL", "V2", "V5"): row_idx = 1
        elif lead in ("III", "aVF", "V3", "V6"): row_idx = 2
        else: row_idx = 3

        manual_baseline = None
        if cfg.manual_baselines and len(cfg.manual_baselines) > row_idx:
            # Absolute image Y baseline minus the crop's top Y coordinate
            manual_baseline = float(cfg.manual_baselines[row_idx] - segment.box[1])

        trace = digitize_lead(segment.crop, calibration.px_per_mm, lead, cfg, manual_baseline)
        traces[lead] = trace
        signals_norm[lead] = trace.signal_norm
        signals_mv[lead] = trace.signal_mv
        metadata[lead] = (trace.baseline_y, trace.peak_px)

    norm_path, mv_path = save_signals_npz(signals_norm, signals_mv, metadata, out, segments, cfg)
    overlay_path = save_npz_overlay(mv_path, norm_path, input_path, calibration.px_per_mm, out, cfg)
    return DigitizationResult(calibration, segments, traces, segment_report, norm_path, mv_path, overlay_path)
