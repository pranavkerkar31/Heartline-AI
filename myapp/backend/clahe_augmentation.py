import cv2
import numpy as np
import os

# -----------------------------
# Paths
# -----------------------------
IMAGE_DIR = "cropped_ecg"
OUTPUT_DIR = "augmented_clahe"

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_files = sorted(
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
)

print(f"Processing {len(image_files)} ECG images...")

# -----------------------------
# CLAHE
# -----------------------------
clahe = cv2.createCLAHE(
    clipLimit=2.2,
    tileGridSize=(8, 8)
)

# -----------------------------
# Detect uneven illumination
# -----------------------------
def has_illumination_problem(gray):
    small = cv2.resize(gray, (200, 200))
    return np.std(small) > 60   # threshold

# -----------------------------
# Illumination correction
# -----------------------------
def illumination_correction(gray):
    blur = cv2.GaussianBlur(gray, (101,101), 0)
    corrected = cv2.divide(gray, blur, scale=255)
    return corrected

# -----------------------------
# Bright background function
# -----------------------------
def brighten_background(v):

    v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)

    alpha = 1.0
    beta = 40
    v = cv2.convertScaleAbs(v, alpha=alpha, beta=beta)

    gamma = 0.6
    v_gamma = np.power(v / 255.0, gamma) * 255
    v_gamma = v_gamma.astype(np.uint8)

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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply illumination correction only if needed
    if has_illumination_problem(gray):
        gray = illumination_correction(gray)

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # CLAHE
    v_clahe = clahe.apply(v)

    # Brighten
    v_final = brighten_background(v_clahe)

    hsv_out = cv2.merge([h, s, v_final])
    output = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)

    name, ext = os.path.splitext(filename)
    out_path = os.path.join(
        OUTPUT_DIR, f"{name}_augmented{ext}"
    )

    cv2.imwrite(out_path, output)

    print(f"✅ {filename} processed")

print("🎉 Done — preprocessing complete.")