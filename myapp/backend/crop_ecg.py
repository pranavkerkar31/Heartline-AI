import cv2
import os

# -------------------- PATHS --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PRED_IMG_DIR = os.path.join(BASE_DIR, "runs", "detect", "predict")
PRED_LABEL_DIR = os.path.join(PRED_IMG_DIR, "labels")
OUTPUT_DIR = os.path.join(BASE_DIR, "predicted_crops")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- SETTINGS --------------------
PADDING = 10  # pixels safety margin

# -------------------- PROCESS IMAGES --------------------
images = [
    f for f in os.listdir(PRED_IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Found {len(images)} predicted images")

for img_name in images:
    img_path = os.path.join(PRED_IMG_DIR, img_name)
    label_path = os.path.join(
        PRED_LABEL_DIR, os.path.splitext(img_name)[0] + ".txt"
    )

    if not os.path.exists(label_path):
        print(f"[SKIP] No prediction label for {img_name}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Cannot read {img_name}")
        continue

    h, w = img.shape[:2]

    # -------------------- READ BOXES --------------------
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            cls, x_c, y_c, bw, bh = map(float, line.strip().split())
            boxes.append((bw * bh, x_c, y_c, bw, bh))

    if not boxes:
        print(f"[SKIP] No boxes in {img_name}")
        continue

    # -------------------- KEEP LARGEST BOX --------------------
    boxes.sort(reverse=True)
    _, x_c, y_c, bw, bh = boxes[0]

    # -------------------- YOLO → PIXEL COORDS --------------------
    x1 = int((x_c - bw / 2) * w)
    y1 = int((y_c - bh / 2) * h)
    x2 = int((x_c + bw / 2) * w)
    y2 = int((y_c + bh / 2) * h)

    # -------------------- PADDING + CLIP --------------------
    x1 = max(0, x1 - PADDING)
    y1 = max(0, y1 - PADDING)
    x2 = min(w, x2 + PADDING)
    y2 = min(h, y2 + PADDING)

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        print(f"[ERROR] Empty crop for {img_name}")
        continue

    # -------------------- SAVE --------------------
    out_name = os.path.splitext(img_name)[0] + "_crop.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    cv2.imwrite(out_path, crop)
    print(f"[OK] Saved {out_name}")

print("\n✅ Cropping from predictions complete.")
