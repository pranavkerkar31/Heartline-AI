import cv2
import numpy as np
import os
import sys

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# 🔹 SET YOUR PATHS HERE
input_dir = "../yolo_ecg/images/test"     # <-- change this
output_dir = "straight_images"   # <-- change this

manual_angle = None     # or set like 2.5 if needed
show_preview = False    # True if you want preview window


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def detect_skew_angle(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scale = min(1.0, 1200 / max(img.shape[:2]))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray.copy()

    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=80,
        minLineLength=int(small.shape[1] * 0.15),
        maxLineGap=20,
    )

    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            angle_deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle_deg) <= 20:
                angles.append(angle_deg)

    if len(angles) >= 5:
        return float(np.median(angles))

    # fallback
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    largest = max(contours, key=cv2.contourArea)
    rect    = cv2.minAreaRect(largest)
    angle   = rect[2]
    if angle < -45:
        angle += 90

    return float(angle)


def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    return cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def deskew_ecg(input_path: str, output_dir: str) -> str:
    img = load_image(input_path)

    if manual_angle is not None:
        skew = manual_angle
    else:
        skew = detect_skew_angle(img)

    correction = -skew

    if abs(correction) < 0.1:
        corrected = img.copy()
    else:
        corrected = rotate_image(img, correction)

    base, ext = os.path.splitext(os.path.basename(input_path))
    out_path = os.path.join(output_dir, f"{base}_deskewed{ext}")
    cv2.imwrite(out_path, corrected)

    # optional preview
    if show_preview:
        cv2.imshow("Corrected", corrected)
        cv2.waitKey(0)

    return out_path


def main():
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    all_files = sorted(os.listdir(input_dir))
    image_files = [
        f for f in all_files
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ]

    if not image_files:
        print("No images found in input folder.")
        sys.exit(0)

    print(f"Processing {len(image_files)} images...")

    for fname in image_files:
        fpath = os.path.join(input_dir, fname)
        try:
            out = deskew_ecg(fpath, output_dir)
            print(f"✔ {fname} → {out}")
        except Exception as e:
            print(f"✖ {fname}: {e}")


if __name__ == "__main__":
    main()