"""Overlay and error visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ecg_digitizer.config import DigitizerConfig
from ecg_digitizer.utils.image import contiguous_runs, ensure_dir


def save_npz_overlay(
    npz_mv_path: str | Path,
    npz_norm_path: str | Path,
    input_path: str | Path,
    px_per_mm: float,
    outdir: str | Path,
    config: DigitizerConfig | None = None,
) -> Path:
    """Overlay mV NPZ traces onto the original ECG image."""
    import cv2
    import matplotlib.pyplot as plt

    cfg = config or DigitizerConfig()
    data_mv = np.load(npz_mv_path)
    data_norm = np.load(npz_norm_path)
    img = cv2.imread(str(input_path))
    if img is None:
        raise FileNotFoundError(f"Could not load: {input_path}")
    overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = overlay.shape[:2]
    green = (0, 220, 80)
    thickness = 1

    for lead in cfg.lead_order:
        key = "RHYTHM" if lead == "Rhythm" else lead
        if key not in data_mv.files:
            continue
        sig_mv = data_mv[key].astype(float)
        box_key = f"{lead}__box"
        if box_key in data_norm.files:
            bx1, by1, bx2, by2 = [int(v) for v in data_norm[box_key]]
        else:
            idx = cfg.lead_order.index(lead)
            row, col = idx // 4, idx % 4
            bx1, bx2 = int(col * w / 4), int((col + 1) * w / 4)
            by1, by2 = (int(row * h * 0.75 / 3), int((row + 1) * h * 0.75 / 3)) if idx < 12 else (int(0.75 * h), h)
        baseline = float(data_norm[f"{lead}__baseline"])
        px_disp = sig_mv * cfg.mm_per_mv * px_per_mm
        y_crop = baseline - px_disp
        crop_h = max(1, by2 - by1)
        _, lead_hi = cfg.amplitude_bounds_mv.get(lead, (0.05, 4.0))
        valid = (
            np.isfinite(y_crop) &
            (np.abs(px_disp) <= (lead_hi / 2.0 + 0.45) * cfg.mm_per_mv * px_per_mm) &
            (y_crop >= -0.18 * crop_h) &
            (y_crop <= crop_h * 1.18)
        )
        n = len(sig_mv)
        left_guard = int(round(cfg.label_skip_for_lead(lead) * n))
        valid[:left_guard] = False
        valid[max(0, n - max(2, int(0.015 * n))):] = False
        x_full = np.linspace(bx1, bx2 - 1, n)
        y_full = by1 + y_crop
        for start, end in contiguous_runs(valid):
            if end - start + 1 < 2:
                continue
            xs = np.clip(x_full[start:end + 1], 0, w - 1).astype(np.int32)
            ys = np.clip(y_full[start:end + 1], 0, h - 1).astype(np.int32)
            pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], isClosed=False, color=green, thickness=thickness, lineType=cv2.LINE_AA)

    out = ensure_dir(outdir) / "ecg_npz_overlay.png"
    fig, ax = plt.subplots(figsize=(22, 10), dpi=120)
    ax.imshow(overlay, aspect="auto", interpolation="lanczos")
    ax.set_title("ECG - mV NPZ traces overlaid  (green = ecg_signals_mv.npz)", fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
