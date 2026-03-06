import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = os.path.join(BASE_DIR, "images", "train")
LABEL_DIR = os.path.join(BASE_DIR, "labels", "train")
OUTPUT_DIR = os.path.join(BASE_DIR, "cropped_ecg")

os.makedirs(OUTPUT_DIR, exist_ok=True)

images = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

for img_name in images:
    label_path = os.path.join(
        LABEL_DIR, os.path.splitext(img_name)[0] + ".txt"
    )

    if not os.path.exists(label_path):
        print(f"Missing label for {img_name}")
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Could not read {img_name}")
        continue

    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    # Handle multiple boxes if present
    for idx, line in enumerate(lines):
        cls, x_c, y_c, bw, bh = map(float, line.strip().split())

        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)

        # Safety clipping
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        cropped = img[y1:y2, x1:x2]

        if cropped.size == 0:
            print(f"Empty crop for {img_name}")
            continue

        out_name = f"{os.path.splitext(img_name)[0]}_crop_{idx}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        cv2.imwrite(out_path, cropped)
        print(f"Saved cropped ECG: {out_name}")
