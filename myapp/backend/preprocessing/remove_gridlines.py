import cv2
import numpy as np
import os

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "runs", "detect", "cropped_ecg")
OUTPUT_DIR = os.path.join(BASE_DIR, "preprocessing", "no_grid_ecg")

os.makedirs(OUTPUT_DIR, exist_ok=True)
# ========================================


def remove_gridlines_preserve_ecg(img):
    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Binarize (adaptive → works for all colors)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5
    )

    # 3. Detect horizontal gridlines
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (40, 1)
    )
    horizontal_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, horizontal_kernel
    )

    # 4. Detect vertical gridlines
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, 40)
    )
    vertical_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, vertical_kernel
    )

    # 5. Combine gridlines
    grid_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)

    # 6. Remove gridlines from binary image
    ecg_only = cv2.subtract(binary, grid_mask)

    # 7. Clean small noise (preserve text + signal)
    kernel = np.ones((2, 2), np.uint8)
    ecg_only = cv2.morphologyEx(ecg_only, cv2.MORPH_OPEN, kernel)

    return ecg_only


def main():
    images = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if not images:
        print("[INFO] No images found.")
        return

    for img_name in images:
        img_path = os.path.join(INPUT_DIR, img_name)
        save_path = os.path.join(OUTPUT_DIR, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"[SKIP] {img_name}")
            continue

        result = remove_gridlines_preserve_ecg(img)
        cv2.imwrite(save_path, result)

        print(f"[OK] Grid removed: {img_name}")

    print("\n✅ Gridline removal completed successfully.")


if __name__ == "__main__":
    main()
