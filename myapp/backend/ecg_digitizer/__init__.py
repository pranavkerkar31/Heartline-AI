"""Research-grade ECG paper digitization toolkit.

The package is a modular version of the notebook pipeline.  It preserves the
notebook's public outputs while making calibration, segmentation, tracing,
validation, visualization, and export independently testable.
"""

from .config import DigitizerConfig, LEAD_NAMES_ORDER
from .pipeline import DigitizationResult, run_digitization

__all__ = [
    "DigitizationResult",
    "DigitizerConfig",
    "LEAD_NAMES_ORDER",
    "run_digitization",
]
