import cv2
import numpy as np
import os

INPUT_DIR = "cropped_ecg"
OUTPUT_DIR = "no_grid_ecg"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_paper(gray):

    background = cv2.medianBlur(gray, 51)

    gray = cv2.divide(gray, background, scale=255)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    return gray

# ---------------------------------------------
# Remove small grid cells using connected comps
# ---------------------------------------------
def remove_small_grid(binary):

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    output = np.zeros_like(binary)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        # keep ECG waveform (long horizontal shapes)
        if area > 400 and width > 50:
            output[labels == i] = 255

    return output


# ---------------------------------------------
# Main grid removal function
# ---------------------------------------------
def remove_ecg_gridlines_safe(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # handle paper photos
    gray = preprocess_paper(gray)

    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    h, w = binary.shape

    # detect long horizontal lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 2, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)

    # detect long vertical lines
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 2))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)

    grid_mask = cv2.bitwise_or(horizontal, vertical)

    protected = binary.copy()
    protected[grid_mask == 255] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    restored = cv2.morphologyEx(protected, cv2.MORPH_CLOSE, kernel, iterations=2)

    # strengthen ECG line
    restored = cv2.dilate(restored, np.ones((2,2),np.uint8), iterations=1)

    # remove grid squares
    final = remove_small_grid(restored)

    return final


# ---------------------------------------------
# Process all ECG images in folder
# ---------------------------------------------
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

        output = remove_ecg_gridlines_safe(img)

        save_name = os.path.splitext(img_name)[0] + ".png"
        save_path = os.path.join(OUTPUT_DIR, save_name)

        cv2.imwrite(save_path, output)

        print(f"✅ {img_name} processed")

    print("🎉 Done.")


if __name__ == "__main__":
    process_folder()