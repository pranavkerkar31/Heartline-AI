from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


LEAD_NAMES_ORDER: tuple[str, ...] = (
    "I", "aVR", "V1", "V4",
    "II", "aVL", "V2", "V5",
    "III", "aVF", "V3", "V6",
    "Rhythm",
)

MENDELEY_P2P_BOUNDS_MV: dict[str, tuple[float, float]] = {
    "I": (0.05, 2.5),
    "II": (0.05, 2.5),
    "III": (0.05, 2.5),
    "aVR": (0.05, 2.0),
    "aVL": (0.05, 2.0),
    "aVF": (0.05, 2.5),
    "V1": (0.05, 3.5),
    "V2": (0.10, 4.0),
    "V3": (0.10, 4.0),
    "V4": (0.10, 4.0),
    "V5": (0.05, 3.0),
    "V6": (0.05, 3.0),
    "Rhythm": (0.05, 4.0),
}


@dataclass(slots=True)
class DigitizerConfig:
    """Tunable parameters for the ECG digitization pipeline."""

    output_dir: Path = Path("outputs/ecg_may")
    mm_per_mv: float = 10.0
    mm_per_sec: float = 25.0
    sample_hz: float = 500.0
    default_px_per_mm: float = 11.8
    lead_order: tuple[str, ...] = LEAD_NAMES_ORDER
    amplitude_bounds_mv: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(MENDELEY_P2P_BOUNDS_MV)
    )

    manual_px_per_mm: float | None = None
    manual_px_per_mm_x: float | None = None
    manual_px_per_mm_y: float | None = None
    manual_baselines: list[float] | None = None

    trace_brightness_min: int = 135
    trace_saturation_max: int = 75
    first_column_label_skip_fraction: float = 0.10
    non_first_label_skip_fraction: float = 0.12
    rhythm_label_skip_fraction: float = 0.06
    edge_label_zone_fraction: float = 0.18

    viterbi_max_gap_mm: float = 1.5
    qrs_rescue_min_height_mm: float = 0.45
    qrs_rescue_max_width_mm: float = 2.0
    max_rescue_amplitude_mv: float = 2.6
    artifact_column_span_fraction: float = 0.46
    artifact_column_density_fraction: float = 0.28

    calibration_min_px_per_mm: float = 5.0
    calibration_max_px_per_mm: float = 25.0
    calibration_regions: tuple[tuple[float, float, float, float], ...] = (
        (0.05, 0.05, 0.95, 0.95),
        (0.10, 0.10, 0.45, 0.45),
        (0.55, 0.10, 0.90, 0.45),
        (0.10, 0.55, 0.45, 0.90),
        (0.55, 0.55, 0.90, 0.90),
    )

    def label_skip_for_lead(self, lead_name: str) -> float:
        """Return the fraction of a crop to ignore at the left edge."""
        if lead_name in {"I", "II", "III"}:
            return self.first_column_label_skip_fraction
        if lead_name == "Rhythm":
            return self.rhythm_label_skip_fraction
        return self.non_first_label_skip_fraction

    def target_samples_for_lead(self, lead_name: str) -> int:
        """Return the Mendeley-compatible sample count for a lead."""
        duration_sec = 10.0 if lead_name == "Rhythm" else 2.5
        return int(round(self.sample_hz * duration_sec))
