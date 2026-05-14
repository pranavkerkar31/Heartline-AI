"""Lead segmentation and OCR-assisted localization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from ecg_digitizer.config import DigitizerConfig
from ecg_digitizer.utils.image import ensure_dir, is_dark_background


_LEAD_TOKENS = {
    "I": re.compile(r"^\|?I$", re.I),
    "II": re.compile(r"^\|?II$", re.I),
    "III": re.compile(r"^\|?III$", re.I),
    "aVR": re.compile(r"a\s*[Vv]\s*[Rr]", re.I),
    "aVL": re.compile(r"a\s*[Vv]\s*[Ll]", re.I),
    "aVF": re.compile(r"a\s*[Vv]\s*[Ff]", re.I),
    "V1": re.compile(r"[Vv]\s*1"),
    "V2": re.compile(r"[Vv]\s*2"),
    "V3": re.compile(r"[Vv]\s*3"),
    "V4": re.compile(r"[Vv]\s*4"),
    "V5": re.compile(r"[Vv]\s*5"),
    "V6": re.compile(r"[Vv]\s*6"),
}


@dataclass(slots=True)
class LeadSegment:
    """One segmented lead crop and its image-space box."""

    name: str
    crop: np.ndarray
    box: tuple[int, int, int, int]
    image_path: Path
    ocr_ok: bool = True


@dataclass(slots=True)
class SegmentReport:
    """Diagnostics from lead segmentation."""

    split_method: str
    row_bands: list[tuple[int, int]]
    ocr_mismatches: list[str]


def _match_lead(text: str) -> str | None:
    token = text.strip()
    for name, pattern in _LEAD_TOKENS.items():
        if pattern.search(token):
            return name
    return None


def binarize_for_traces(img: np.ndarray) -> np.ndarray:
    """Otsu binarization with foreground in white."""
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(th) > 127:
        th = 255 - th
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)
    return (clean > 127).astype(np.uint8) * 255


def horizontal_segments(binary: np.ndarray, min_h_ratio: float = 0.03) -> list[tuple[int, int]]:
    """Find contiguous horizontal bands with foreground ECG content."""
    h, w = binary.shape
    proj = np.convolve(np.sum(binary > 0, axis=1).astype(float), np.ones(15) / 15, mode="same")
    active = proj > max(3, 0.02 * w)
    segs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_active in enumerate(active):
        if is_active and start is None:
            start = idx
        elif not is_active and start is not None:
            segs.append((start, idx))
            start = None
    if start is not None:
        segs.append((start, h))
    min_h = int(min_h_ratio * h)
    return [(s, e) for s, e in segs if e - s >= min_h]


def _ocr_lead_labels(img: np.ndarray, row_segs: list[tuple[int, int]]) -> list[dict[str, int | str]]:
    """Run optional Tesseract OCR on row header strips."""
    import cv2
    from PIL import Image
    import pytesseract

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results: list[dict[str, int | str]] = []
    for row_idx, (y1, y2) in enumerate(row_segs[:3]):
        strip_h = max(8, int(0.30 * (y2 - y1)))
        strip = gray[y1: y1 + strip_h, :]
        scale = 3
        strip_up = cv2.resize(strip, (strip.shape[1] * scale, strip.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        _, strip_bin = cv2.threshold(strip_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(strip_bin) < 127:
            strip_bin = 255 - strip_bin
        try:
            data = pytesseract.image_to_data(
                Image.fromarray(strip_bin),
                config="--psm 11 --oem 3 -c tessedit_char_whitelist=IVaAvVRrLlFf123456",
                output_type=pytesseract.Output.DICT,
            )
        except Exception:
            return results
        for i, text in enumerate(data["text"]):
            if not text.strip():
                continue
            try:
                conf = int(float(data["conf"][i]))
            except Exception:
                conf = -1
            if conf < 20:
                continue
            lead = _match_lead(text)
            if lead is None:
                continue
            results.append({
                "lead": lead,
                "x_center": (data["left"][i] + data["width"][i] // 2) // scale,
                "row": row_idx,
            })
    return results


def find_column_splits(img: np.ndarray, row_segs: list[tuple[int, int]], desired: int = 4) -> tuple[list[int], str]:
    """Find column boundaries using OCR, separator lines, then equal-width fallback."""
    import cv2
    from scipy.signal import find_peaks

    h, w = img.shape[:2]
    equal = [int(i * w / desired) for i in range(desired + 1)]

    def valid(xs: list[int], min_frac: float) -> bool:
        return len(xs) == desired + 1 and all(xs[i + 1] - xs[i] >= min_frac * w for i in range(len(xs) - 1))

    try:
        hits = _ocr_lead_labels(img, row_segs)
    except Exception:
        hits = []
    if hits:
        expected = [int((i + 0.5) * w / desired) for i in range(desired)]
        buckets: list[list[int]] = [[] for _ in range(desired)]
        for hit in hits:
            x = int(hit["x_center"])
            buckets[int(np.argmin([abs(x - e) for e in expected]))].append(x)
        anchors = [int(np.median(b)) if b else None for b in buckets]
        known = [(i, v) for i, v in enumerate(anchors) if v is not None]
        if len(known) >= 2:
            for i in range(desired):
                if anchors[i] is None:
                    anchors[i] = int(np.interp(i, [k[0] for k in known], [k[1] for k in known]))
        if all(a is not None for a in anchors):
            xs = [0] + [int((anchors[i] + anchors[i + 1]) / 2) for i in range(desired - 1)] + [w]  # type: ignore[index]
            if valid(xs, 0.08):
                return xs, "OCR"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row_mask = np.zeros(h, dtype=bool)
    for y1, y2 in row_segs[:3]:
        row_mask[y1:y2] = True
    brightness = gray.astype(np.float32)
    brightness[~row_mask, :] = 0
    col_mean = brightness.sum(axis=0) / max(1, row_mask.sum())
    smooth = np.convolve(col_mean, np.ones(3) / 3, mode="same")
    positive = smooth[smooth > 0]
    if positive.size:
        threshold = np.percentile(positive, 85)
        peaks, _ = find_peaks(smooth, height=threshold, distance=max(10, w // (desired * 2)), prominence=threshold * 0.2)
        good = []
        for col in peaks:
            frac = np.mean(gray[row_mask, int(col)] > 180)
            if frac >= 0.30:
                good.append(int(col))
        if len(good) >= desired - 1:
            expected = [int((i + 1) * w / desired) for i in range(desired - 1)]
            chosen: list[int] = []
            remaining = list(good)
            for e in expected:
                idx = int(np.argmin(np.abs(np.asarray(remaining) - e)))
                chosen.append(remaining.pop(idx))
            xs = [0] + sorted(chosen) + [w]
            if valid(xs, 0.08):
                return xs, "white-separator"

    return equal, "equal-width"


def _tight_waveform_bbox(gray: np.ndarray, min_row_density: float = 0.005) -> tuple[int, int]:
    """Return y bounds around likely waveform ink while ignoring lead labels."""
    h, w = gray.shape
    mask = gray.copy().astype(np.float32)
    lh, lw = int(0.35 * h), int(0.12 * w)
    mask[:lh, :lw] = np.nan
    flat = mask[~np.isnan(mask)]
    if flat.size == 0:
        return h // 4, 3 * h // 4
    dark_bg = is_dark_background(gray)
    threshold = np.percentile(flat, 85 if dark_bg else 15)
    fg = (mask >= threshold) if dark_bg else (mask <= threshold)
    fg[:lh, :lw] = False
    row_counts = fg.sum(axis=1)
    dense = np.where((row_counts >= max(1, int(min_row_density * w))) & (row_counts <= int(0.35 * w)))[0]
    if len(dense) == 0:
        return h // 4, 3 * h // 4
    return int(dense[0]), int(dense[-1])


def _crop_and_save(img: np.ndarray, outdir: Path, box: tuple[int, int, int, int], name: str) -> LeadSegment:
    """Crop a lead and save it for debugging/reproducibility."""
    import cv2

    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    crop = img[y1:y2, x1:x2].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    top, bot = _tight_waveform_bbox(gray)
    py = max(8, int(0.08 * max(1, y2 - y1)))
    px = max(2, int(0.015 * max(1, x2 - x1)))
    fy1, fy2 = max(0, y1 + top - py), min(h, y1 + bot + py)
    fx1, fx2 = max(0, x1 - px), min(w, x2 + px)
    final = img[fy1:fy2, fx1:fx2].copy()
    path = outdir / f"Lead_{name}.png"
    cv2.imwrite(str(path), final)
    return LeadSegment(name=name, crop=final, box=(fx1, fy1, fx2, fy2), image_path=path)


def segment_leads(img: np.ndarray, outdir: str | Path, config: DigitizerConfig | None = None) -> tuple[dict[str, LeadSegment], SegmentReport]:
    """Segment 12 diagnostic leads plus rhythm strip."""
    cfg = config or DigitizerConfig()
    out = ensure_dir(outdir)
    h, w = img.shape[:2]
    binary = binarize_for_traces(img)
    rows = horizontal_segments(binary, min_h_ratio=0.03)
    if len(rows) < 3:
        rows = horizontal_segments(binary, min_h_ratio=0.02)
    if len(rows) < 3:
        sh = h // 4
        rows = [(i * sh, (i + 1) * sh) for i in range(4)]
    rows = sorted(rows, key=lambda item: item[0])
    while len(rows) > 4:
        gaps = [rows[i + 1][0] - rows[i][1] for i in range(len(rows) - 1)]
        idx = int(np.argmin(gaps))
        rows = rows[:idx] + [(rows[idx][0], rows[idx + 1][1])] + rows[idx + 2:]

    diag_rows = rows[:3]
    rhythm_rows = rows[3:] if len(rows) > 3 else [(int(0.75 * h), h)]
    if len(diag_rows) == 3:
        centers = [int((y1 + y2) / 2) for y1, y2 in diag_rows]
        gap = max(1, int(np.median(np.diff(centers))))
        bounds = [max(0, centers[0] - int(0.45 * gap))]
        bounds.extend(int((centers[i] + centers[i + 1]) / 2) for i in range(2))
        rhythm_center = int((rhythm_rows[0][0] + rhythm_rows[0][1]) / 2)
        bounds.append(int((centers[-1] + rhythm_center) / 2))
        diag_rows = [(max(0, bounds[i]), min(h, bounds[i + 1])) for i in range(3)]
        rhythm_rows = [(bounds[-1], h)]

    xs, method = find_column_splits(img, diag_rows, desired=4)
    boxes = [(xs[c], y1, xs[c + 1], y2) for y1, y2 in diag_rows for c in range(len(xs) - 1)]
    boxes.extend((0, y1, w, y2) for y1, y2 in rhythm_rows)
    boxes = (boxes + [boxes[-1]])[:len(cfg.lead_order)]

    segments: dict[str, LeadSegment] = {}
    for name, box in zip(cfg.lead_order, boxes):
        segments[name] = _crop_and_save(img, out, box, name)
    return segments, SegmentReport(split_method=method, row_bands=diag_rows + rhythm_rows, ocr_mismatches=[])
