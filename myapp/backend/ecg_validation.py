import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks, resample_poly


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6", "RHYTHM"]
LEAD_ALIASES = {"RYTHM": "RHYTHM"}


class ValidationPaths(object):
    def __init__(self, dataset_root, uploads_root, run_id, category, record_number):
        self.dataset_root = Path(dataset_root)
        self.uploads_root = Path(uploads_root)
        self.run_id = run_id
        self.category = category
        self.record_number = int(record_number)


def _canonical_lead(lead):
    return LEAD_ALIASES.get(lead, lead)


def _fill_nan(x):
    x = np.asarray(x, dtype=float)
    if np.isfinite(x).all():
        return x
    good = np.flatnonzero(np.isfinite(x))
    if len(good) == 0:
        return np.zeros_like(x, dtype=float)
    filled = x.copy()
    bad = np.flatnonzero(~np.isfinite(x))
    filled[bad] = np.interp(bad, good, x[good])
    return filled


def _robust_z(x, eps=1e-9):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if scale < eps:
        scale = np.std(x) + eps
    return (x - med) / scale


def _baseline_correct(x, fs):
    x = _fill_nan(x)
    win = max(3, int(round(0.20 * fs)))
    if win % 2 == 0:
        win += 1
    baseline = median_filter(x, size=win, mode="nearest")
    return x - baseline


def _alignment_feature(x, fs):
    centered = _baseline_correct(x, fs)
    z = _robust_z(centered)
    z = np.clip(z, -6, 6)
    dz = np.diff(z, prepend=z[0])
    energy = np.abs(z) + 0.8 * np.abs(dz)
    smooth_win = max(3, int(round(0.045 * fs)))
    energy = uniform_filter1d(energy, size=smooth_win, mode="nearest")
    return _robust_z(energy)


def _resample_if_needed(x, src_fs, dst_fs):
    if abs(src_fs - dst_fs) < 1e-6:
        return np.asarray(x, dtype=float)
    from fractions import Fraction

    ratio = Fraction(dst_fs / src_fs).limit_denominator(1000)
    return resample_poly(np.asarray(x, dtype=float), ratio.numerator, ratio.denominator)


def _detect_anchor_peaks(feature, fs):
    distance = max(1, int(round(0.25 * fs)))
    height = np.percentile(feature, 70)
    prominence = max(0.25, np.percentile(feature, 85) - np.percentile(feature, 50))
    peaks, _ = find_peaks(feature, height=height, distance=distance, prominence=prominence)
    return peaks


def _normalized_overlap_score(truth_feat, paper_feat, offset, scale):
    truth_idx = np.arange(len(truth_feat), dtype=float)
    paper_pos = (truth_idx - offset) / scale
    valid = (paper_pos >= 0) & (paper_pos <= len(paper_feat) - 1)
    if valid.sum() < max(80, 0.15 * min(len(truth_feat), len(paper_feat))):
        return -np.inf, valid, None

    p_interp = np.interp(paper_pos[valid], np.arange(len(paper_feat)), paper_feat)
    g = truth_feat[valid] - np.mean(truth_feat[valid])
    p = p_interp - np.mean(p_interp)
    denom = np.std(g) * np.std(p)
    if denom < 1e-6:
        return -np.inf, valid, p_interp
    score = float(np.mean((g / np.std(g)) * (p / np.std(p))))
    return score, valid, p_interp


def _anchor_match_score(truth_peaks, paper_peaks, offset, scale, fs):
    if len(truth_peaks) == 0 or len(paper_peaks) == 0:
        return 0.0, 0

    mapped = offset + scale * paper_peaks
    tol = max(8, int(round(0.08 * fs)))
    matched_truth = set()
    score = 0.0

    for mapped_peak in mapped:
        distances = np.abs(truth_peaks - mapped_peak)
        j = int(np.argmin(distances))
        d = float(distances[j])
        if d <= tol and j not in matched_truth:
            matched_truth.add(j)
            score += math.exp(-0.5 * (d / tol) ** 2)

    n_matches = len(matched_truth)
    if n_matches < 2:
        score *= 0.20
    return float(score), n_matches


def _candidate_offsets_from_peaks(truth_peaks, paper_peaks, scale, n_truth, n_paper, step):
    offsets = []
    if len(truth_peaks) and len(paper_peaks):
        for tp in truth_peaks:
            for pp in paper_peaks:
                offsets.append(tp - scale * pp)

    min_offset = -scale * n_paper + 0.20 * min(n_truth, n_paper)
    max_offset = n_truth - 0.20 * min(n_truth, n_paper)
    offsets.extend(np.arange(min_offset, max_offset + step, step))
    return np.asarray(offsets, dtype=float)


def _estimate_horizontal_alignment(truth, paper, fs, scale_min=0.95, scale_max=1.05):
    truth_feat = _alignment_feature(truth, fs)
    paper_feat = _alignment_feature(paper, fs)
    truth_peaks = _detect_anchor_peaks(truth_feat, fs)
    paper_peaks = _detect_anchor_peaks(paper_feat, fs)

    best = {
        "score": -np.inf,
        "corr_score": -np.inf,
        "anchor_score": 0.0,
        "anchor_matches": 0,
        "offset": 0.0,
        "scale": 1.0,
        "valid": np.zeros(len(truth_feat), dtype=bool),
        "truth_peaks": truth_peaks,
        "paper_peaks": paper_peaks,
    }

    offset_step = max(5, int(round(0.025 * fs)))
    scales = np.arange(scale_min, scale_max + 0.001, 0.002)

    for scale in scales:
        offsets = _candidate_offsets_from_peaks(truth_peaks, paper_peaks, scale, len(truth_feat), len(paper_feat), offset_step)
        offsets = np.unique(np.round(offsets / offset_step) * offset_step)
        for offset in offsets:
            corr_score, valid, _ = _normalized_overlap_score(truth_feat, paper_feat, offset, scale)
            anchor_score, anchor_matches = _anchor_match_score(truth_peaks, paper_peaks, offset, scale, fs)
            score = anchor_score + 0.35 * max(corr_score, -1.0)
            if score > best["score"]:
                best.update(
                    score=score,
                    corr_score=corr_score,
                    anchor_score=anchor_score,
                    anchor_matches=anchor_matches,
                    offset=float(offset),
                    scale=float(scale),
                    valid=valid,
                )

    return best


def _robust_vertical_fit(truth, paper_on_truth, valid):
    mask = valid & np.isfinite(truth) & np.isfinite(paper_on_truth)
    if mask.sum() < 10:
        return 1.0, 0.0, mask

    x_all = paper_on_truth[mask]
    y_all = truth[mask]
    ax = np.abs(x_all - np.median(x_all))
    ay = np.abs(y_all - np.median(y_all))
    active = (ax >= np.percentile(ax, 65)) & (ay >= np.percentile(ay, 65))
    active &= ax <= np.percentile(ax, 98)
    active &= ay <= np.percentile(ay, 98)
    if active.sum() < 20:
        active = (ax >= np.percentile(ax, 75)) | (ay >= np.percentile(ay, 75))

    x = x_all[active]
    y = y_all[active]
    if len(x) < 10:
        return 1.0, 0.0, mask

    keep = np.ones_like(x, dtype=bool)
    gain = 1.0
    offset = 0.0
    for _ in range(4):
        design = np.column_stack([x[keep], np.ones(keep.sum())])
        gain, offset = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        resid = y - (gain * x + offset)
        cutoff = np.percentile(np.abs(resid), 80)
        keep = np.abs(resid) <= max(cutoff, 1e-9)

    active_positions = np.flatnonzero(mask)[active]
    full_keep = np.zeros_like(mask)
    full_keep[active_positions[keep]] = True
    return float(gain), float(offset), full_keep


def _align_lead(truth_raw, paper_raw, fs_truth, fs_paper):
    paper_raw = _resample_if_needed(paper_raw, fs_paper, fs_truth)
    truth_raw = _fill_nan(truth_raw)
    paper_raw = _fill_nan(paper_raw)
    truth_centered = _baseline_correct(truth_raw, fs_truth)
    paper_centered = _baseline_correct(paper_raw, fs_truth)

    horiz = _estimate_horizontal_alignment(truth_centered, paper_centered, fs_truth)
    truth_idx = np.arange(len(truth_centered), dtype=float)
    paper_pos = (truth_idx - horiz["offset"]) / horiz["scale"]
    valid = (paper_pos >= 0) & (paper_pos <= len(paper_centered) - 1)

    paper_on_truth = np.full(len(truth_centered), np.nan)
    paper_on_truth[valid] = np.interp(paper_pos[valid], np.arange(len(paper_centered)), paper_centered)

    gain, v_offset, fit_mask = _robust_vertical_fit(truth_centered, paper_on_truth, valid)
    paper_aligned = gain * paper_on_truth + v_offset
    residual = truth_centered[fit_mask] - paper_aligned[fit_mask]
    rmse = float(np.sqrt(np.mean(residual ** 2))) if fit_mask.sum() else float("nan")

    return {
        "truth_centered": truth_centered,
        "paper_centered": paper_centered,
        "paper_aligned": paper_aligned,
        "valid": valid,
        "fit_mask": fit_mask,
        "offset_seconds": horiz["offset"] / fs_truth,
        "offset_samples": horiz["offset"],
        "scale": horiz["scale"],
        "score": horiz["score"],
        "corr_score": horiz["corr_score"],
        "anchor_score": horiz["anchor_score"],
        "anchor_matches": horiz["anchor_matches"],
        "gain": gain,
        "vertical_offset": v_offset,
        "rmse": rmse,
    }


def _validation_metrics(result):
    truth = result["truth_centered"]
    paper = result["paper_aligned"]
    mask = result["fit_mask"].copy()
    if mask.sum() < 10:
        mask = result["valid"] & np.isfinite(truth) & np.isfinite(paper)

    y_true = truth[mask]
    y_pred = paper[mask]
    if len(y_true) < 2:
        return {"samples": int(len(y_true)), "pcc": float("nan"), "rmse_mv": float("nan"), "snr_db": float("nan")}

    error = y_true - y_pred
    rmse = float(np.sqrt(np.mean(error ** 2)))
    true_std = np.std(y_true)
    pred_std = np.std(y_pred)
    pcc = float(np.corrcoef(y_true, y_pred)[0, 1]) if true_std >= 1e-12 and pred_std >= 1e-12 else float("nan")
    signal_power = float(np.mean(y_true ** 2))
    noise_power = float(np.mean(error ** 2))
    snr_db = float("inf") if noise_power < 1e-12 else float(10 * np.log10(signal_power / noise_power))
    return {"samples": int(len(y_true)), "pcc": pcc, "rmse_mv": rmse, "snr_db": snr_db}


def _dataset_file(dataset_root, category, record_number):
    dataset_category = dataset_root / category
    if not dataset_category.exists():
        raise IOError("Category folder not found: {}".format(dataset_category))

    filename = "{}({}).npz".format(category, record_number)
    npz_path = dataset_category / filename
    if not npz_path.exists():
        raise IOError("Dataset file not found: {}".format(npz_path))
    return npz_path


def _sanitize_float(value):
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _plot_validation(leads, lead_results, fs_truth, output_path):
    fig, axes = plt.subplots(len(leads), 2, figsize=(18, 4 * len(leads)), squeeze=False)

    for row, lead in enumerate(leads):
        result = lead_results[lead]
        truth_centered = result["truth_centered"]
        paper_centered = result["paper_centered"]
        paper_aligned = result["paper_aligned"]
        fit_mask = result["fit_mask"]

        t_truth = np.arange(len(truth_centered)) / fs_truth
        t_paper = np.arange(len(paper_centered)) / fs_truth

        ax_left = axes[row, 0]
        ax_left.plot(t_truth, truth_centered, color="black", lw=1.4, alpha=0.75, label="Ground truth")
        ax_left.plot(t_paper, paper_centered, color="red", lw=1.2, alpha=0.75, label="Paper extracted")
        ax_left.set_title("Lead {} - original / unaligned".format(lead), fontweight="bold")
        ax_left.set_ylabel("Voltage (mV, baseline-corrected)")
        ax_left.grid(True, linestyle="--", alpha=0.5)
        ax_left.legend(loc="best")

        ax_right = axes[row, 1]
        ax_right.plot(t_truth, truth_centered, color="black", lw=1.4, alpha=0.65, label="Ground truth")
        ax_right.plot(t_truth, paper_aligned, color="blue", lw=1.2, alpha=0.80, label="Paper aligned")
        if fit_mask.any():
            ymin = np.nanmin(np.vstack([truth_centered, paper_aligned])) - 0.03
            ymax = np.nanmax(np.vstack([truth_centered, paper_aligned])) + 0.03
            ax_right.fill_between(
                t_truth,
                ymin,
                ymax,
                where=fit_mask,
                color="tab:green",
                alpha=0.08,
                step="mid",
                label="samples used for vertical fit",
            )

        lag_ms = 1000 * result["offset_seconds"]
        ax_right.set_title(
            "Lead {} - robust aligned | offset={:.0f} ms, scale={:.3f}, score={:.2f}".format(
                lead, lag_ms, result["scale"], result["score"]
            ),
            fontweight="bold",
        )
        ax_right.set_ylabel("Voltage (mV, aligned)")
        ax_right.grid(True, linestyle="--", alpha=0.5)
        ax_right.legend(loc="best")

    axes[-1, 0].set_xlabel("Time (seconds)")
    axes[-1, 1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def validate_signals(extracted_path, paths):
    extracted_path = Path(extracted_path)
    truth_path = _dataset_file(paths.dataset_root, paths.category, paths.record_number)

    extr_data = np.load(str(extracted_path))
    truth_data = np.load(str(truth_path))

    fs_truth = float(truth_data["sampling_rate"]) if "sampling_rate" in truth_data.files else 500.0
    fs_extr = float(extr_data["sampling_rate"]) if "sampling_rate" in extr_data.files else fs_truth

    shared_leads = []
    for lead in LEADS:
        canonical = _canonical_lead(lead)
        if canonical in truth_data.files and canonical in extr_data.files:
            shared_leads.append(canonical)

    lead_results = {}
    lead_metrics = []
    export_rows = []

    for lead in shared_leads:
        result = _align_lead(truth_data[lead].astype(float), extr_data[lead].astype(float), fs_truth, fs_extr)
        metrics = _validation_metrics(result)
        lead_results[lead] = result
        lead_metrics.append(
            {
                "lead": lead,
                "your_pcc": _sanitize_float(metrics["pcc"]),
                "your_rmse": _sanitize_float(metrics["rmse_mv"]),
                "your_snr": _sanitize_float(metrics["snr_db"]),
                "offset_ms": _sanitize_float(1000 * result["offset_seconds"]),
                "time_scale": _sanitize_float(result["scale"]),
                "match_score": _sanitize_float(result["score"]),
                "samples": int(metrics["samples"]),
            }
        )
        export_rows.append(
            {
                "category": paths.category,
                "record_number": paths.record_number,
                "lead": lead,
                "your_pcc": _sanitize_float(metrics["pcc"]),
                "your_rmse": _sanitize_float(metrics["rmse_mv"]),
                "your_snr": _sanitize_float(metrics["snr_db"]),
            }
        )

    validation_dir = paths.uploads_root / "validation" / paths.run_id
    validation_dir.mkdir(parents=True, exist_ok=True)
    plot_path = validation_dir / "comparison.png"
    _plot_validation(shared_leads, lead_results, fs_truth, plot_path)

    mean_pcc = [row["your_pcc"] for row in lead_metrics if row["your_pcc"] is not None]
    mean_rmse = [row["your_rmse"] for row in lead_metrics if row["your_rmse"] is not None]
    mean_snr = [row["your_snr"] for row in lead_metrics if row["your_snr"] is not None]

    return {
        "category": paths.category,
        "record_number": paths.record_number,
        "truth_npz_path": str(truth_path),
        "comparison_image": str(plot_path),
        "lead_metrics": lead_metrics,
        "export_rows": export_rows,
        "summary": {
            "lead_count": len(lead_metrics),
            "mean_pcc": _sanitize_float(float(np.mean(mean_pcc))) if mean_pcc else None,
            "mean_rmse": _sanitize_float(float(np.mean(mean_rmse))) if mean_rmse else None,
            "mean_snr": _sanitize_float(float(np.mean(mean_snr))) if mean_snr else None,
        },
    }


def parse_record_number(value):
    if isinstance(value, int):
        return value
    match = re.search(r"(\d+)", str(value))
    if not match:
        raise ValueError("Invalid record number: {}".format(value))
    return int(match.group(1))
