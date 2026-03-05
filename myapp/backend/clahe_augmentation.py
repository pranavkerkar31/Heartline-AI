import cv2
import numpy as np
import os

# -----------------------------
# Paths
# -----------------------------
IMAGE_DIR = "cropped_ecg"                 # input images (paper + digital)
OUTPUT_DIR = "augmented_clahe"  # output images

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_files = sorted(
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
)

print(f"Processing {len(image_files)} ECG images...")

# -----------------------------
# CLAHE setup (stronger for paper ECG)
# -----------------------------
clahe = cv2.createCLAHE(
    clipLimit=2.2,
    tileGridSize=(8, 8)
)

# -----------------------------
# Bright background function
# -----------------------------
def brighten_background(v):
    # Normalize brightness
    v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)

    # Increase brightness (key step)
    alpha = 1.0   # keep contrast stable
    beta = 40     # brightness boost
    v = cv2.convertScaleAbs(v, alpha=alpha, beta=beta)

    # Gamma correction (<1 makes image brighter)
    gamma = 0.6
    v_gamma = np.power(v / 255.0, gamma) * 255
    v_gamma = v_gamma.astype(np.uint8)

    # Push background to white
    v_gamma[v_gamma > 230] = 255

    return v_gamma


# -----------------------------
# Process images
# -----------------------------
for filename in image_files:
    img_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Skipping {filename}")
        continue

    # BGR → HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # CLAHE on brightness
    v_clahe = clahe.apply(v)

    # Brighten background
    v_final = brighten_background(v_clahe)

    # Merge back
    hsv_out = cv2.merge([h, s, v_final])
    output = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)

    # Save
    name, ext = os.path.splitext(filename)
    out_path = os.path.join(
        OUTPUT_DIR, f"{name}_augmented{ext}"
    )
    cv2.imwrite(out_path, output)

    print(f"✅ {filename} processed")

print("🎉 Done — paper ECG brightness now matches digital ECG.")