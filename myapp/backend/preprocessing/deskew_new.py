"""
ECG Image Deskew Preprocessor  (v3 — Red Corner Detection)
============================================================
Straightens phone-camera-captured ECG paper images so that YOLO bounding
box detection works correctly.

Detection priority:
  1. RED CORNER MARKS  — The ECG paper has 4 red marks at each corner of the
                          grid border. Detect these precisely and use them as
                          control points for a perspective warp. Most accurate.

  2. PAPER BOUNDARY    — Detect the white paper sheet against the dark table
                          background using brightness thresholding + contour
                          finding. Good fallback for clean phone photos.

  3. FINE ROTATION     — After any warp, measure residual tilt from dominant
                          horizontal ECG grid lines and correct sub-degree skew.

  4. HOUGH FALLBACK    — Last resort: rotate based on dominant Hough line angle.

Usage:
    python ecg_deskew.py          # uses hardcoded INPUT_DIR / OUTPUT_DIR
"""

import cv2
import numpy as np
import sys
from pathlib import Path


# ── Configure paths here ──────────────────────────────────────────────────────
INPUT_DIR  = "../yolo_ecg/crop_dataset/images/val"
OUTPUT_DIR = "output"
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def order_points(pts):
    """Order 4 corner points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]     # top-left
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[2] = pts[np.argmax(s)]     # bottom-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def four_point_transform(image, pts):
    """Warp perspective so the 4 pts become a perfect rectangle."""
    rect = order_points(pts)
    tl, tr, br, bl = rect
    max_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    max_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([
        [0,         0        ],
        [max_w - 1, 0        ],
        [max_w - 1, max_h - 1],
        [0,         max_h - 1],
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_w, max_h))


def is_valid_warp(shape):
    """Reject wildly non-rectangular warp results."""
    h, w = shape[:2]
    return max(w, h) / max(min(w, h), 1) <= 5


# ══════════════════════════════════════════════════════════════════════════════
# Method 1 — Red corner mark detection (PRIMARY)
# ══════════════════════════════════════════════════════════════════════════════

def detect_red_corner_marks(image):
    """
    Detect the 4 red corner marks printed on the ECG paper border.

    In phone photos the red marks appear with high R-channel dominance
    (R > 120, R > B*1.4, R > G*1.4). We find the 4 blobs, take their
    centroids, and return them as corner control points.

    Returns float32 (4, 2) array of [TL, TR, BR, BL] corners, or None.
    """
    b_ch, g_ch, r_ch = cv2.split(image)

    # Isolate red-dominant pixels
    red_mask = (
        (r_ch.astype(np.int32) > 120) &
        (r_ch.astype(np.int32) > b_ch.astype(np.int32) * 1.4) &
        (r_ch.astype(np.int32) > g_ch.astype(np.int32) * 1.4)
    ).astype(np.uint8) * 255

    # Morphological closing to group nearby pixels into blobs
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (5,  5))
    closed  = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k_close)
    cleaned = cv2.morphologyEx(closed,   cv2.MORPH_OPEN,  k_open)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 4:
        return None

    # Sort by area descending, keep top 4 (the 4 corner marks)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

    centers = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        centers.append(rect[0])   # (cx, cy)

    if len(centers) != 4:
        return None

    return np.array(centers, dtype="float32")


# ══════════════════════════════════════════════════════════════════════════════
# Method 2 — Paper boundary detection (FALLBACK)
# ══════════════════════════════════════════════════════════════════════════════

def detect_paper_corners(image):
    """
    Detect the white ECG paper sheet against a dark background.

    Uses brightness threshold 180 (more reliable than Otsu on phone photos)
    and tries multiple approxPolyDP epsilons to get exactly 4 corners.

    Returns float32 (4, 2) array or None.
    """
    h, w = image.shape[:2]
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

    k      = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  k)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours     = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        if cv2.contourArea(cnt) < 0.10 * h * w:
            continue
        peri = cv2.arcLength(cnt, True)
        for eps in (0.01, 0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype("float32")
        rect = cv2.minAreaRect(cnt)
        box  = cv2.boxPoints(rect)
        return box.astype("float32")

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Method 3 — Fine rotation correction (POST-PROCESS)
# ══════════════════════════════════════════════════════════════════════════════

def fine_tune_rotation(image, max_angle: float = 5.0):
    """
    Remove sub-degree residual tilt by measuring dominant horizontal line
    angle from the ECG grid after perspective correction.

    Returns (corrected_image, angle_applied).
    """
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 120, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=80, minLineLength=80, maxLineGap=15,
    )
    if lines is None:
        return image, 0.0

    angles = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x2 == x1:
            continue
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) < max_angle:
            angles.append(a)

    if not angles:
        return image, 0.0

    angle = float(np.median(angles))
    if abs(angle) < 0.2:
        return image, 0.0

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    corrected = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return corrected, angle


# ══════════════════════════════════════════════════════════════════════════════
# Method 4 — Hough rotation fallback (LAST RESORT)
# ══════════════════════════════════════════════════════════════════════════════

def hough_rotation_correct(image, max_angle: float = 15.0):
    """Fallback: rotate based on dominant Hough line angle."""
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
    if lines is None:
        return image, 0.0

    angles = []
    for line in lines:
        rho, theta = line[0]
        a = np.degrees(theta) - 90
        if abs(a) <= max_angle:
            angles.append(a)

    if not angles:
        return image, 0.0

    angle = float(np.median(angles))
    if abs(angle) < 0.5:
        return image, 0.0

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def deskew_ecg(image_path: str, debug: bool = False):
    """
    Load an ECG image and return a perfectly straightened version.

    Pipeline (in priority order):
      1. Detect 4 red corner marks  -> perspective warp  [most accurate]
      2. Detect paper boundary      -> perspective warp  [good fallback]
      3. Fine rotation correction   -> sub-degree fix    [always applied]
      4. Hough rotation             -> last resort

    Parameters
    ----------
    image_path : path to the input image
    debug      : if True, saves overlay images showing detected corners

    Returns
    -------
    (np.ndarray, str) — deskewed BGR image, method label
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    method = "none"
    warped = img

    # ── 1. Red corner mark detection ─────────────────────────────────────────
    red_corners = detect_red_corner_marks(img)

    if red_corners is not None:
        if debug:
            dbg = img.copy()
            for pt in red_corners:
                cv2.circle(dbg, (int(pt[0]), int(pt[1])), 15, (0, 255, 0), 3)
            cv2.imwrite(image_path.replace(".", "_debug_red."), dbg)

        candidate = four_point_transform(img, red_corners)
        if is_valid_warp(candidate.shape):
            warped = candidate
            method = "red_corners"

    # ── 2. Paper boundary detection (if red corners not found) ───────────────
    if method == "none":
        paper_corners = detect_paper_corners(img)
        if paper_corners is not None:
            if debug:
                dbg = img.copy()
                for pt in paper_corners:
                    cv2.circle(dbg, (int(pt[0]), int(pt[1])), 12, (255, 0, 0), -1)
                cv2.imwrite(image_path.replace(".", "_debug_paper."), dbg)

            candidate = four_point_transform(img, paper_corners)
            if is_valid_warp(candidate.shape):
                warped = candidate
                method = "paper_boundary"

    # ── 3. Fine rotation correction (always applied after any warp) ──────────
    corrected, fine_angle = fine_tune_rotation(warped)
    if abs(fine_angle) >= 0.2:
        warped = corrected
        method = (method + "+fine_rotation") if method != "none" else "fine_rotation"

    if method != "none":
        return warped, method

    # ── 4. Hough rotation fallback ───────────────────────────────────────────
    fallback, fb_angle = hough_rotation_correct(img)
    if abs(fb_angle) >= 0.5:
        return fallback, "hough_rotation"

    return img, "none"


# ══════════════════════════════════════════════════════════════════════════════
# Batch processing
# ══════════════════════════════════════════════════════════════════════════════

def process_path(input_path: str, output_dir: str, debug: bool = False):
    p       = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = [f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_EXT]
    else:
        print(f"ERROR: {input_path} does not exist.")
        sys.exit(1)

    for f in files:
        try:
            result, method = deskew_ecg(str(f), debug=debug)
            out_path = out_dir / f.name
            cv2.imwrite(str(out_path), result)
            print(f"[{method:35s}]  {f.name}  ->  {out_path}")
        except Exception as e:
            print(f"[ERROR]  {f.name}: {e}")


def main():
    process_path(INPUT_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()