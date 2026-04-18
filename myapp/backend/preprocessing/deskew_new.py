
import cv2
import numpy as np
import os
import shutil

# ─────────────────────────── CONFIG ───────────────────────────
INPUT_DIR  = "../yolo_ecg/crop_dataset/images/val"
OUTPUT_DIR = "output"
SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")

os.makedirs(INPUT_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  DETECTION: Is this a digital/scanned image or a camera photo?
# ══════════════════════════════════════════════════════════════

def is_digital_image(img):
    """
    Classifies the image as DIGITAL (scan/export) or CAMERA PHOTO.

    Digital ECGs have:
      • Very uniform, bright (white) background outside the ECG box
      • No dark background regions (table, floor, shadow)
      • Straight pixel-perfect edges
      • Aspect ratio already landscape (w > h) in almost all cases

    Camera photos have:
      • Uneven lighting / shadows
      • Non-white background
      • Perspective distortion

    Strategy: sample the four outer border strips (5% of each side).
    A scan will have nearly pure white there; a camera photo won't.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    bh = max(4, int(h * 0.05))
    bw = max(4, int(w * 0.05))

    top    = gray[:bh, :]
    bottom = gray[h - bh:, :]
    left   = gray[:, :bw]
    right  = gray[:, w - bw:]

    border = np.concatenate([top.ravel(), bottom.ravel(),
                             left.ravel(), right.ravel()])

    mean_b = float(np.mean(border))
    std_b  = float(np.std(border))

    # Fraction of pixels that are clearly dark (background clutter)
    dark_ratio = float(np.sum(gray < 60)) / gray.size

    # Fraction of border pixels that are near-white
    white_ratio = float(np.sum(border > 220)) / len(border)

    digital = (mean_b > 195) and (std_b < 30) and \
              (dark_ratio < 0.04) and (white_ratio > 0.75)

    print(f"    [classify] mean={mean_b:.1f}  std={std_b:.1f}  "
          f"dark={dark_ratio:.3f}  white={white_ratio:.2f}"
          f"  → {'DIGITAL' if digital else 'CAMERA'}")
    return digital


# ══════════════════════════════════════════════════════════════
#  CAMERA PIPELINE — Step 1: Perspective warp
# ══════════════════════════════════════════════════════════════

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s       = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff    = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect
    maxW = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    maxH = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    dst  = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
    M    = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


def detect_document_corners(img):
    h, w  = img.shape[:2]
    scale = 800 / max(h, w)
    small = cv2.resize(img, (int(w*scale), int(h*scale)))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try 3 strategies, use first that yields a quad
    thresholds = []
    _, t1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholds.append(t1)
    t2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    thresholds.append(t2)
    t3 = cv2.dilate(cv2.Canny(blur, 50, 200), np.ones((3,3), np.uint8))
    thresholds.append(t3)

    min_area = small.shape[0] * small.shape[1] * 0.10
    for thresh in thresholds:
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts[:5]:
            if cv2.contourArea(c) < min_area:
                continue
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2) / scale

    margin = int(min(h, w) * 0.03)
    return np.array([[margin,margin],[w-margin,margin],
                     [w-margin,h-margin],[margin,h-margin]], dtype=np.float32)


# ══════════════════════════════════════════════════════════════
#  CAMERA PIPELINE — Step 2: Orientation fix (0/90/180/270)
# ══════════════════════════════════════════════════════════════

def orientation_score(gray):
    """
    Score how likely this is the correct upright ECG orientation.
    ECG header (patient info) sits at the top → brighter top strip.
    Grid lines are horizontal → more horizontal Hough lines.
    """
    h, w = gray.shape
    top_bright = np.mean(gray[:int(h*0.12), :])
    bot_bright = np.mean(gray[int(h*0.88):, :])
    score      = float(top_bright - bot_bright)

    blur  = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                             minLineLength=int(w*0.15), maxLineGap=20)
    if lines is not None:
        for ln in lines:
            x1,y1,x2,y2 = ln[0]
            if x2!=x1 and abs(np.degrees(np.arctan2(y2-y1,x2-x1))) < 10:
                score += 0.5
    return score


def fix_orientation(img):
    """
    Pick the best of 4 rotations.
    Only rotates if the best orientation scores clearly higher than 0°.
    Gap threshold = 4.0 to avoid touching already-correct images.
    """
    rots = [
        (0,   img),
        (90,  cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
        (180, cv2.rotate(img, cv2.ROTATE_180)),
        (270, cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]
    scored = [(orientation_score(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)), d, r)
              for d, r in rots]
    scored.sort(reverse=True)

    best_s, best_d, best_r = scored[0]
    orig_s = next(s for s,d,_ in scored if d == 0)
    gap    = best_s - orig_s

    print(f"    [orient] best={best_d}deg  gap={gap:.2f}")
    if best_d == 0 or gap < 4.0:
        print(f"    [orient] keeping original (gap too small)")
        return img
    print(f"    [orient] rotating to {best_d}deg")
    return best_r


# ══════════════════════════════════════════════════════════════
#  CAMERA PIPELINE — Step 3: Fine skew (Hough, ±15°)
# ══════════════════════════════════════════════════════════════

def detect_fine_skew(img):
    """
    Detect and return the correction angle for residual small tilt.
    Only acts if ≥5 confident near-horizontal lines agree AND |angle| ≥ 0.4°.
    Returns 0.0 to mean "do nothing".
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 1200 / max(img.shape[:2]))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    blur  = cv2.GaussianBlur(small, (5,5), 0)
    edges = cv2.Canny(blur, 40, 120, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi/360, 60,
                             minLineLength=int(small.shape[1]*0.20),
                             maxLineGap=15)
    if lines is None:
        return 0.0

    angles = []
    for ln in lines:
        x1,y1,x2,y2 = ln[0]
        if x2 == x1:
            continue
        a = np.degrees(np.arctan2(y2-y1, x2-x1))
        if abs(a) <= 15:
            angles.append(a)

    if len(angles) < 5:
        return 0.0

    median = float(np.median(angles))
    if abs(median) < 0.4:
        return 0.0

    print(f"    [skew]  detected={median:.2f}deg  correction={-median:.2f}deg")
    return -median


def rotate_image(img, angle_deg):
    h, w   = img.shape[:2]
    cx, cy = w/2, h/2
    M      = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos    = abs(M[0,0]); sin = abs(M[0,1])
    nw     = int(h*sin + w*cos)
    nh     = int(h*cos + w*sin)
    M[0,2] += nw/2 - cx
    M[1,2] += nh/2 - cy
    return cv2.warpAffine(img, M, (nw, nh),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255,255,255))


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def process_image(inp, out):
    print(f"\n  {'─'*52}")
    print(f"  File : {os.path.basename(inp)}")

    img = cv2.imread(inp)
    if img is None:
        print(f"  ERROR: Cannot read file.")
        return False

    print(f"  Size : {img.shape[1]} x {img.shape[0]} px")

    if is_digital_image(img):
        # ── DIGITAL: copy as-is, no transformation at all ──
        print(f"  Mode : DIGITAL — copying unchanged")
        shutil.copy2(inp, out)
        print(f"  Saved: {out}")
        return True

    # ── CAMERA PHOTO: full pipeline ──
    print(f"  Mode : CAMERA — perspective warp + orientation + fine skew")

    corners  = detect_document_corners(img)
    warped   = four_point_transform(img, corners)
    print(f"  Warp : {warped.shape[1]} x {warped.shape[0]} px")

    oriented = fix_orientation(warped)

    angle = detect_fine_skew(oriented)
    final = rotate_image(oriented, angle) if abs(angle) >= 0.4 else oriented

    cv2.imwrite(out, final)
    print(f"  Saved: {out}")
    return True


# ─────────────────────────── RUN ───────────────────────────

print(f"\n{'═'*55}")
print(f"  ECG Deskew Pipeline")
print(f"{'═'*55}")
print(f"  Input  : {os.path.abspath(INPUT_DIR)}")
print(f"  Output : {os.path.abspath(OUTPUT_DIR)}")
print(f"{'═'*55}")

files = sorted(f for f in os.listdir(INPUT_DIR)
               if f.lower().endswith(SUPPORTED_EXTS))

if not files:
    print(f"\n  No images found in '{INPUT_DIR}/'")
    print(f"  Supported : {', '.join(SUPPORTED_EXTS)}")
    print(f"  → Drop ECG images into the input/ folder and run again.\n")
    exit()

print(f"\n  Found {len(files)} image(s) ...")

ok = fail = 0
for fname in files:
    inp  = os.path.join(INPUT_DIR, fname)
    name, ext = os.path.splitext(fname)
    out  = os.path.join(OUTPUT_DIR, f"{name}_deskewed{ext}")
    if process_image(inp, out):
        ok += 1
    else:
        fail += 1

print(f"\n{'═'*55}")
print(f"  Done — {ok} OK, {fail} failed")
print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")
print(f"{'═'*55}\n")