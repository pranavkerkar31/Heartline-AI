import cv2
import os

# Paths
IMAGE_DIR = "b"
OUTPUT_DIR = "augmented_clahe"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load image names
image_files = sorted(
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
)

print(f"Processing {len(image_files)} ECG images...")

# CLAHE setup
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for filename in image_files:
    img_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Skipping {filename}")
        continue

    # BGR → HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # CLAHE on brightness channel
    v_clahe = clahe.apply(v)

    # Global normalization
    v_norm = cv2.normalize(
        v_clahe, None, 0, 255, cv2.NORM_MINMAX
    )

    # Merge back
    hsv_aug = cv2.merge([h, s, v_norm])
    augmented = cv2.cvtColor(hsv_aug, cv2.COLOR_HSV2BGR)

    # Save
    name, ext = os.path.splitext(filename)
    out_path = os.path.join(
        OUTPUT_DIR, f"{name}_augmented{ext}"
    )
    cv2.imwrite(out_path, augmented)

print("✅ ECG brightness augmentation completed.")
