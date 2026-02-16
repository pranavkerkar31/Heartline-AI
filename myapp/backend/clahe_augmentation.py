import cv2
import numpy as np
import os

# ================= PATHS =================
IMAGE_DIR = "runs/detect/cropped_ecg"
OUTPUT_DIR = "brightness_augmented"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= PARAMETERS =================
TARGET_BG_BRIGHTNESS = 200   # Mendeley-like paper brightness
KERNEL_SIZE = 51             # Background estimation scale
MIN_GAIN = 0.9               # Prevent over-brightening
MAX_GAIN = 1.6               # Prevent blowout

# ================= PROCESSING FUNCTION =================
def normalize_ecg_background(img):
    """
    Normalize ECG paper background brightness without
    affecting ECG waveforms or contrast.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Estimate background illumination
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE)
    )
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # Compute average background brightness (ignore dark ECG lines)
    bg_pixels = background[background > 30]
    if len(bg_pixels) == 0:
        return img  # safety fallback

    current_bg = np.mean(bg_pixels)

    # Compute adaptive gain
    gain = TARGET_BG_BRIGHTNESS / (current_bg + 1e-6)
    gain = np.clip(gain, MIN_GAIN, MAX_GAIN)

    # Apply illumination normalization
    normalized = gray.astype(np.float32) * gain
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    # Return as BGR
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

# ================= BATCH PROCESS =================
image_files = sorted(
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
)

print(f"📂 Processing {len(image_files)} ECG images...")

for filename in image_files:
    img_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠️ Skipping unreadable file: {filename}")
        continue

    enhanced = normalize_ecg_background(img)

    name, ext = os.path.splitext(filename)
    out_path = os.path.join(
        OUTPUT_DIR, f"{name}_bg_normalized{ext}"
    )

    cv2.imwrite(out_path, enhanced)

print("✅ ECG background normalization completed.")
