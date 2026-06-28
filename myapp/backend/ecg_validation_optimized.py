"""
Optimized ECG validation with parallel processing and performance improvements.
"""

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import concurrent.futures

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks, resample_poly

# Reuse constants and helper functions from original module
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6", "RHYTHM"]
LEAD_ALIASES = {"RYTHM": "RHYTHM"}

class ValidationPaths(object):
    def __init__(self, dataset_root, uploads_root, run_id, category, record_number):
        self.dataset_root = Path(dataset_root)
        self.uploads_root = Path(uploads_root)
        self.run_id = run_id
        self.category = category
        self.record_number = int(record_number)

# Reuse helper functions from original module
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
    return (x - med) / (scale + eps)

def _sanitize_float(x):
    if x is None or math.isnan(x) or math.isinf(x):
        return None
    return float(x)

# Optimized alignment function with reduced computational complexity
def _align_lead_optimized(truth_raw, paper_raw, fs_truth, fs_paper):
    """Optimized version of lead alignment with faster algorithms."""
    
    # Faster resampling with linear interpolation
    if fs_paper != fs_truth:
        duration = len(paper_raw) / fs_paper
        target_samples = int(duration * fs_truth)
        paper_raw = np.interp(
            np.linspace(0, len(paper_raw)-1, target_samples),
            np.arange(len(paper_raw)),
            paper_raw
        )
    
    truth_raw = _fill_nan(truth_raw)
    paper_raw = _fill_nan(paper_raw)
    
    # Faster baseline correction using moving average
    def _baseline_correct_fast(signal, fs):
        window = max(1, int(0.2 * fs))  # 200ms window
        baseline = uniform_filter1d(signal, size=window)
        return signal - baseline
    
    truth_centered = _baseline_correct_fast(truth_raw, fs_truth)
    paper_centered = _baseline_correct_fast(paper_raw, fs_truth)
    
    # Simplified horizontal alignment using cross-correlation
    def _estimate_horizontal_alignment_fast(truth, paper, fs):
        # Use downsampled signals for faster correlation
        step = max(1, int(fs / 100))  # 10ms steps
        truth_ds = truth[::step]
        paper_ds = paper[::step]
        
        # Simple cross-correlation for alignment
        n = len(truth_ds)
        m = len(paper_ds)
        if m > n:
            paper_ds = paper_ds[:n]
            m = n
        elif n > m:
            truth_ds = truth_ds[:m]
            n = m
        
        # Find best offset using simple correlation
        correlations = []
        offsets = range(-int(fs/10), int(fs/10))  # ±100ms range
        
        for offset in offsets:
            if offset >= 0:
                truth_seg = truth_ds[offset:offset+m]
                paper_seg = paper_ds[:m]
            else:
                truth_seg = truth_ds[:m]
                paper_seg = paper_ds[-offset:-offset+m]
            
            if len(truth_seg) == len(paper_seg) and len(truth_seg) > 0:
                corr = np.corrcoef(truth_seg, paper_seg)[0, 1]
                correlations.append((offset, corr))
        
        if correlations:
            best_offset, best_corr = max(correlations, key=lambda x: x[1])
            offset_samples = best_offset * step
            scale = 1.0  # Assume no scaling for speed
            score = best_corr
        else:
            offset_samples = 0
            scale = 1.0
            score = 0.0
        
        return {
            "offset": offset_samples,
            "scale": scale,
            "score": float(score),
            "corr_score": float(score),
            "anchor_score": 0.0,
            "anchor_matches": 0
        }
    
    horiz = _estimate_horizontal_alignment_fast(truth_centered, paper_centered, fs_truth)
    
    # Simplified alignment
    offset_samples = horiz["offset"]
    scale = horiz["scale"]
    
    # Create aligned paper signal
    truth_idx = np.arange(len(truth_centered), dtype=float)
    paper_pos = (truth_idx - offset_samples) / scale
    valid = (paper_pos >= 0) & (paper_pos < len(paper_centered))
    
    paper_on_truth = np.full(len(truth_centered), np.nan)
    if valid.any():
        paper_on_truth[valid] = np.interp(
            paper_pos[valid], 
            np.arange(len(paper_centered)), 
            paper_centered
        )
    
    # Fast vertical fitting
    valid_mask = valid & np.isfinite(truth_centered) & np.isfinite(paper_on_truth)
    if valid_mask.sum() > 10:  # Minimum samples for reliable fit
        # Simple linear regression
        X = paper_on_truth[valid_mask].reshape(-1, 1)
        y = truth_centered[valid_mask]
        
        # Fast calculation without full SVD
        X_mean = X.mean()
        y_mean = y.mean()
        covariance = ((X - X_mean) * (y - y_mean)).mean()
        var_X = ((X - X_mean) ** 2).mean()
        
        if var_X > 1e-6:
            gain = covariance / var_X
        else:
            gain = 1.0
        
        v_offset = y_mean - gain * X_mean
        
        # Simple RMSE calculation
        residual = truth_centered[valid_mask] - (gain * paper_on_truth[valid_mask] + v_offset)
        rmse = float(np.sqrt(np.mean(residual ** 2)))
    else:
        gain = 1.0
        v_offset = 0.0
        rmse = float("nan")
    
    return {
        "truth_centered": truth_centered,
        "paper_centered": paper_centered,
        "paper_aligned": gain * paper_on_truth + v_offset,
        "valid": valid,
        "fit_mask": valid_mask,
        "offset_seconds": offset_samples / fs_truth,
        "offset_samples": offset_samples,
        "scale": scale,
        "score": horiz["score"],
        "corr_score": horiz["corr_score"],
        "anchor_score": horiz["anchor_score"],
        "anchor_matches": horiz["anchor_matches"],
        "gain": gain,
        "vertical_offset": v_offset,
        "rmse": rmse,
    }

# Optimized metrics calculation
def _validation_metrics_optimized(result):
    """Faster metrics calculation with simplified algorithms."""
    truth = result["truth_centered"]
    paper = result["paper_aligned"]
    valid = result["fit_mask"]
    
    if not valid.any():
        return {
            "pcc": None,
            "rmse_mv": None,
            "snr_db": None,
            "samples": 0,
        }
    
    # Fast PCC calculation
    truth_valid = truth[valid]
    paper_valid = paper[valid]
    
    if len(truth_valid) < 10:
        return {
            "pcc": None,
            "rmse_mv": None,
            "snr_db": None,
            "samples": len(truth_valid),
        }
    
    # Pearson correlation coefficient (fast version)
    truth_centered = truth_valid - truth_valid.mean()
    paper_centered = paper_valid - paper_valid.mean()
    
    if np.sqrt((truth_centered ** 2).sum()) > 0 and np.sqrt((paper_centered ** 2).sum()) > 0:
        pcc = (truth_centered * paper_centered).sum() / (
            np.sqrt((truth_centered ** 2).sum()) * np.sqrt((paper_centered ** 2).sum())
        )
    else:
        pcc = 0.0
    
    # Use pre-calculated RMSE if available
    rmse = result["rmse"] if not math.isnan(result["rmse"]) else None
    
    # Fast SNR estimation
    if rmse is not None and rmse > 0:
        signal_power = np.mean(truth_valid ** 2)
        noise_power = rmse ** 2
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float("inf")
    else:
        snr = None
    
    return {
        "pcc": float(pcc),
        "rmse_mv": float(rmse) if rmse is not None else None,
        "snr_db": float(snr) if snr is not None else None,
        "samples": len(truth_valid),
    }

# Optimized main validation function with parallel processing
def validate_signals_optimized(extracted_path, paths):
    """Optimized validation with parallel lead processing."""
    extracted_path = Path(extracted_path)
    
    # Reuse dataset file function (would need to be imported or duplicated)
    def _dataset_file(dataset_root, category, record_number):
        # Simplified version - would need full implementation
        return dataset_root / category / f"{record_number}.npz"
    
    truth_path = _dataset_file(paths.dataset_root, paths.category, paths.record_number)
    
    # Load data
    extr_data = np.load(str(extracted_path))
    truth_data = np.load(str(truth_path))
    
    fs_truth = float(truth_data["sampling_rate"]) if "sampling_rate" in truth_data.files else 500.0
    fs_extr = float(extr_data["sampling_rate"]) if "sampling_rate" in extr_data.files else fs_truth
    
    # Find shared leads
    shared_leads = []
    for lead in LEADS:
        canonical = _canonical_lead(lead)
        if canonical in truth_data.files and canonical in extr_data.files:
            shared_leads.append(canonical)
    
    if not shared_leads:
        return {"error": "No shared leads found"}
    
    # Parallel processing of leads
    lead_results = {}
    lead_metrics = []
    export_rows = []
    
    def process_lead(lead):
        """Process a single lead - called in parallel."""
        try:
            result = _align_lead_optimized(
                truth_data[lead].astype(float), 
                extr_data[lead].astype(float), 
                fs_truth, 
                fs_extr
            )
            metrics = _validation_metrics_optimized(result)
            
            return {
                "lead": lead,
                "result": result,
                "metrics": metrics
            }
        except Exception as e:
            return {
                "lead": lead,
                "error": str(e)
            }
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(shared_leads))) as executor:
        futures = {executor.submit(process_lead, lead): lead for lead in shared_leads}
        
        for future in concurrent.futures.as_completed(futures):
            lead_result = future.result()
            lead = lead_result["lead"]
            
            if "error" in lead_result:
                continue
                
            result = lead_result["result"]
            metrics = lead_result["metrics"]
            
            lead_results[lead] = result
            lead_metrics.append({
                "lead": lead,
                "your_pcc": _sanitize_float(metrics["pcc"]),
                "your_rmse": _sanitize_float(metrics["rmse_mv"]),
                "your_snr": _sanitize_float(metrics["snr_db"]),
                "offset_ms": _sanitize_float(1000 * result["offset_seconds"]),
                "time_scale": _sanitize_float(result["scale"]),
                "match_score": _sanitize_float(result["score"]),
                "samples": int(metrics["samples"]),
            })
            export_rows.append({
                "category": paths.category,
                "record_number": paths.record_number,
                "lead": lead,
                "your_pcc": _sanitize_float(metrics["pcc"]),
                "your_rmse": _sanitize_float(metrics["rmse_mv"]),
                "your_snr": _sanitize_float(metrics["snr_db"]),
            })
    
    # Create validation directory and simple comparison plot
    validation_dir = paths.uploads_root / "validation" / paths.run_id
    validation_dir.mkdir(parents=True, exist_ok=True)
    plot_path = validation_dir / "comparison.png"
    
    # Simple comparison plot (faster than detailed one)
    plt.figure(figsize=(12, 8))
    for i, lead in enumerate(shared_leads[:4]):  # Only plot first 4 leads for speed
        plt.subplot(2, 2, i+1)
        if lead in lead_results:
            result = lead_results[lead]
            plt.plot(result["truth_centered"], label=f"Truth {lead}", alpha=0.7)
            plt.plot(result["paper_aligned"], label=f"Extracted {lead}", alpha=0.7)
            plt.title(f"Lead {lead}")
            plt.legend()
    plt.tight_layout()
    plt.savefig(str(plot_path))
    plt.close()
    
    return {
        "category": paths.category,
        "record_number": paths.record_number,
        "truth_npz_path": str(truth_path),
        "comparison_image": str(plot_path),
        "lead_metrics": lead_metrics,
        "export_rows": export_rows,
        "summary": {
            "lead_count": len(shared_leads),
            "mean_pcc": np.mean([m["your_pcc"] for m in lead_metrics if m["your_pcc"] is not None]) if lead_metrics else None,
            "mean_rmse": np.mean([m["your_rmse"] for m in lead_metrics if m["your_rmse"] is not None]) if lead_metrics else None,
            "mean_snr": np.mean([m["your_snr"] for m in lead_metrics if m["your_snr"] is not None]) if lead_metrics else None,
        }
    }