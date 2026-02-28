import cv2
import numpy as np
import os

# ======================================================
# PATHS
# ======================================================
INPUT_DIR = "cropped_ecg"          # ORIGINAL images only
OUTPUT_DIR = "final_ecg"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# CLAHE SETUP
# ======================================================
clahe = cv2.createCLAHE(
    clipLimit=2.2,
    tileGridSize=(8, 8)
)

# ======================================================
# GRIDLINE REMOVAL (SAFE FOR PAPER ECG)
# ======================================================
def remove_ecg_gridlines_safe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Suppress paper texture
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Gentle threshold
    bw = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 3
    )

    # Grid-scale kernels
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))

    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    vertical   = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

    # 🔥 CONNECT broken gridlines
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, connect_kernel)
    vertical   = cv2.morphologyEx(vertical,   cv2.MORPH_CLOSE, connect_kernel)

    # Slight thickening
    thick_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    horizontal = cv2.dilate(horizontal, thick_kernel, iterations=1)
    vertical   = cv2.dilate(vertical,   thick_kernel, iterations=1)

    # Combine grid mask
    grid_mask = cv2.bitwise_or(horizontal, vertical)

    # Remove grid from grayscale
    no_grid = gray.copy()
    no_grid[grid_mask > 0] = 255

    return no_grid

# ======================================================
# BRIGHT BACKGROUND (DIGITAL-LIKE LOOK)
# ======================================================
def brighten_background(v):
    # Normalize
    v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)

    # Brightness lift
    alpha = 1.0
    beta = 40
    v = cv2.convertScaleAbs(v, alpha=alpha, beta=beta)

    # Gamma correction (<1 brightens)
    gamma = 0.7
    v = np.power(v / 255.0, gamma) * 255
    v = v.astype(np.uint8)

    # Push background toward white
    v[v > 230] = 255

    return v

# ======================================================
# PROCESS FOLDER
# ======================================================
def process_folder():
    images = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(f"Processing {len(images)} ECG images...")

    for img_name in images:
        img_path = os.path.join(INPUT_DIR, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # ----------------------------------
        # STEP 1: Grid removal (SAFE)
        # ----------------------------------
        no_grid_gray = remove_ecg_gridlines_safe(img)

        # Convert back to BGR for HSV
        no_grid_bgr = cv2.cvtColor(no_grid_gray, cv2.COLOR_GRAY2BGR)

        # ----------------------------------
        # STEP 2: CLAHE + bright background
        # ----------------------------------
        hsv = cv2.cvtColor(no_grid_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        v = clahe.apply(v)
        v = brighten_background(v)

        hsv_out = cv2.merge([h, s, v])
        output = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)

        # ----------------------------------
        # SAVE
        # ----------------------------------
        name, ext = os.path.splitext(img_name)
        save_path = os.path.join(OUTPUT_DIR, f"{name}_final{ext}")
        cv2.imwrite(save_path, output)

        print(f"✅ {img_name} processed")

    print("🎉 DONE — ECG images processed correctly.")

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    process_folder()