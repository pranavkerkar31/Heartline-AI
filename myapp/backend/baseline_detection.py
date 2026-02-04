import cv2
import numpy as np
import os
import math

INPUT_DIR = "no_grid_ecg"        # ✅ grid-removed images
DEBUG_DIR = "baseline_debug"     # visual verification

os.makedirs(DEBUG_DIR, exist_ok=True)


# ================= BASELINE DETECTION =================
def detect_ecg_baselines(binary):
    """
    Detect ECG baselines using Hough Transform.
    Returns list of baseline y-coordinates.
    """
    h, w = binary.shape

    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=150,
        minLineLength=int(0.8 * w),      # ≥80% width
        maxLineGap=int(0.15 * w)         # merge broken lines
    )

    baselines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))

            # ±2.5° horizontal constraint
            if angle <= 2.5:
                baselines.append((y1 + y2) // 2)

    # ---- Merge nearby baselines (same ECG row) ----
    baselines = sorted(baselines)
    merged = []

    for y in baselines:
        if not merged or abs(merged[-1] - y) > int(0.02 * h):
            merged.append(y)

    return merged


# ================= PROCESS FOLDER =================
def process_folder():
    images = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not images:
        print("❌ No images found in no_grid_ecg/")
        return

    for img_name in images:
        path = os.path.join(INPUT_DIR, img_name)
        binary = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if binary is None:
            continue

        baselines = detect_ecg_baselines(binary)

        # ---- Debug visualization ----
        debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        for y in baselines:
            cv2.line(debug, (0, y), (debug.shape[1], y), (0, 0, 255), 1)

        cv2.imwrite(
            os.path.join(DEBUG_DIR, img_name),
            debug
        )

        print(f"✅ {img_name} | Baselines detected: {len(baselines)}")

    print("\n🎉 Baseline detection completed.")


if __name__ == "__main__":
    process_folder()
