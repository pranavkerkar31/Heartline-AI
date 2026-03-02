
import cv2
import numpy as np
import os

INPUT_DIR = "cropped_ecg"
OUTPUT_DIR = "no_grid_ecg"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def remove_ecg_gridlines_safe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    h, w = binary.shape

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 2, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 2))

    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)

    grid_mask = cv2.bitwise_or(horizontal, vertical)

    protected = binary.copy()
    protected[grid_mask == 255] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    restored = cv2.morphologyEx(protected, cv2.MORPH_CLOSE, kernel, iterations=2)

    final = np.zeros_like(restored)
    final[restored > 0] = 255

    return final


def process_folder():
    images = [f for f in os.listdir(INPUT_DIR)
              if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for img_name in images:
        img_path = os.path.join(INPUT_DIR, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        output = remove_ecg_gridlines_safe(img)

        save_name = os.path.splitext(img_name)[0] + ".png"
        save_path = os.path.join(OUTPUT_DIR, save_name)

        cv2.imwrite(save_path, output)
        print(f"✅ {img_name} processed")

    print("🎉 Done.")


if __name__ == "__main__":
    process_folder()