import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ================= PATHS =================
INPUT_DIR = "C:/Academics/FOURTH YEAR/FINAL YEAR PROJECT/Project2025_26/ECG_dataset/augmented_clahe"
OUTPUT_DIR = "C:/Academics/FOURTH YEAR/FINAL YEAR PROJECT/Project2025_26/ECG_dataset/augmented_agcwd"
NUM_SAMPLES = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= AGCWD FUNCTION =================

def apply_agc_wd(image, alpha=0.5):
    """
    Adaptive Gamma Correction with Weighting Distribution (AGCWD)
    Supports color images by operating on the Value (V) channel in HSV space.
    """

    # Check if image is color
    is_color = len(image.shape) == 3

    if is_color:
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # Use V channel for processing
        process_chan = v
    else:
        process_chan = image.copy()

    # Normalize to [0, 1]
    img_float = process_chan.astype(np.float32) / 255.0

    # Calculate PDF and CDF
    hist, _ = np.histogram(process_chan.flatten(), 256, [0, 256])
    pdf = hist / hist.sum()
    cdf = np.cumsum(pdf)

    # Calculate Weighting Distribution
    wd = cdf ** alpha
    
    # Calculate Adaptive Gamma
    # Use pixel intensities as indices into wd array
    gamma = 1 - wd[process_chan]
    gamma = np.clip(gamma, 0.01, 10.0)

    # Apply Gamma Correction
    enhanced_float = np.power(img_float, gamma)
    enhanced_uint8 = np.clip(enhanced_float * 255.0, 0, 255).astype(np.uint8)

    # Reconstruct image
    if is_color:
        # Merge H, S with enhanced V
        enhanced_hsv = cv2.merge((h, s, enhanced_uint8))
        # Convert back to BGR
        enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    else:
        enhanced_image = enhanced_uint8

    return enhanced_image


# ================= LOAD & AUGMENT =================
original_images = []
augmented_images = []
filenames = []

image_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

image_files = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(image_exts)
]

print(f"📂 Found {len(image_files)} ECG images")

for fname in image_files:
    path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(path)

    if img is None:
        print(f"⚠️ Skipping unreadable image: {fname}")
        continue

    aug = apply_agc_wd(img)

    original_images.append(img)
    augmented_images.append(aug)
    filenames.append(fname)

    cv2.imwrite(os.path.join(OUTPUT_DIR, fname), aug)

print(f"✅ Saved {len(augmented_images)} augmented images")


# ================= VISUALIZATION =================
num_samples = min(NUM_SAMPLES, len(original_images))
idxs = np.random.choice(len(original_images), num_samples, replace=False)

fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 4))
fig.suptitle("Original vs AGCWD-Augmented ECG Images", fontsize=16)

for i, idx in enumerate(idxs):
    orig_rgb = cv2.cvtColor(original_images[idx], cv2.COLOR_BGR2RGB)

    axes[i, 0].imshow(orig_rgb)
    axes[i, 0].set_title("Original")
    axes[i, 0].axis("off")

    # Handle color/grayscale display for augmented image
    aug_img = augmented_images[idx]
    if len(aug_img.shape) == 3:
        aug_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
        axes[i, 1].imshow(aug_rgb)
    else:
        axes[i, 1].imshow(aug_img, cmap="gray")

    axes[i, 1].set_title("AGCWD Augmented")
    axes[i, 1].axis("off")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
