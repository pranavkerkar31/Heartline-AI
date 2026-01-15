import cv2
import numpy as np
import os

def remove_gridlines_strict(img):
    """
    Produces:
    - White background
    - Black ECG waveform
    - Black lead names
    - NO gridlines
    """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Strong blur to suppress grid texture
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binary inverse threshold:
    # Dark content (ECG + text) -> white
    # Background + grid -> black
    _, binary = cv2.threshold(
        blur, 140, 255, cv2.THRESH_BINARY_INV
    )

    # Remove thin grid remnants using morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Fill gaps in ECG waveform
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Convert to final format:
    # ECG -> black, background -> white
    final = cv2.bitwise_not(clean)

    return final


# ---------------- MAIN ----------------
if __name__ == "__main__":

    INPUT_DIR = "runs/detect/cropped_ecg"
    OUTPUT_DIR = "runs/detect/clean_ecg"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for name in images:
        img_path = os.path.join(INPUT_DIR, name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[SKIP] {name}")
            continue

        result = remove_gridlines_strict(img)

        out_path = os.path.join(OUTPUT_DIR, name)
        cv2.imwrite(out_path, result)

        print(f"Clean ECG saved: {name}")

    print("\nAll ECGs cleaned successfully.")
