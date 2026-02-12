import cv2
import numpy as np
import os

from remove_gridlines import remove_ecg_gridlines_safe

INPUT_DIR = "runs/detect/cropped_ecg"
OUTPUT_DIR = "runs/detect/no_grid_ecg"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------- Paper vs Digital Detection ----------------
def is_paper_ecg(gray):
    # Paper ECGs have higher background variance
    return np.std(gray) > 25


def normalize_paper_ecg(gray):
    # Remove paper background & weaken grid
    background = cv2.GaussianBlur(gray, (51, 51), 0)
    normalized = cv2.subtract(gray, background)
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    return normalized


# ---------------- Pipeline ----------------
def process_folder():
    images = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for img_name in images:
        img_path = os.path.join(INPUT_DIR, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Convert to grayscale ONCE
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adaptive preprocessing
        if is_paper_ecg(gray):
            print(f"📄 Paper ECG: {img_name}")
            gray = normalize_paper_ecg(gray)
        else:
            print(f"💻 Digital ECG: {img_name}")

        # Common grid removal
        output = remove_ecg_gridlines_safe(gray)

        save_name = os.path.splitext(img_name)[0] + ".png"
        save_path = os.path.join(OUTPUT_DIR, save_name)

        cv2.imwrite(save_path, output)
        print(f"✅ {img_name} processed")

    print("\n🎉 Grid removal pipeline completed.")


if __name__ == "__main__":
    process_folder()
