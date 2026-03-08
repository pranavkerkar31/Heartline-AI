# cropping only the ecg region
import cv2
import numpy as np
import os

PREDICT_DIR = "../runs/detect/predict2"
OUTPUT_DIR = "cropped_ecg"

LOWER_BLUE = np.array([100, 150, 50])
UPPER_BLUE = np.array([140, 255, 255])

BOX_MARGIN = 12          # removes blue border
TOP_TEXT_RATIO = 0.04   # removes YOLO label text (8%)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def crop_ecg_from_image(image_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read {image_path}")
        return

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Detect blue bounding box
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    # Strengthen box lines
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print(f"[SKIP] No blue box found in {image_path}")
        return

    # Largest contour = ECG box
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Inner crop (remove blue border)
    x1 = max(x + BOX_MARGIN, 0)
    y1 = max(y + BOX_MARGIN, 0)
    x2 = min(x + w - BOX_MARGIN, img.shape[1])
    y2 = min(y + h - BOX_MARGIN, img.shape[0])

    cropped = img[y1:y2, x1:x2]

    # 🔑 Remove YOLO label text strip from top
    top_cut = int(TOP_TEXT_RATIO * cropped.shape[0])
    cropped = cropped[top_cut:, :]

    cv2.imwrite(save_path, cropped)
    print(f"[OK] Saved clean ECG: {save_path}")

def main():
    images = [
        f for f in os.listdir(PREDICT_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if not images:
        print("No images found.")
        return

    for img_name in images:
        input_path = os.path.join(PREDICT_DIR, img_name)
        output_path = os.path.join(OUTPUT_DIR, img_name)
        crop_ecg_from_image(input_path, output_path)

    print("\n ECG waveform extraction completed successfully.")

if __name__ == "__main__":
    main()