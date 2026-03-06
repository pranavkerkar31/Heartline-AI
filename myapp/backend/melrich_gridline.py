import os
import cv2
import numpy as np

INPUT_DIR = "cropped_ecg"
OUTPUT_DIR = "no_grid_ecg"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===== SKELETONIZE =====
def skeletonize(img):
    img = img.copy()
    skel = np.zeros(img.shape, np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            break

    return skel


# ===== FFT GRID REMOVAL =====
def fft_remove_grid(gray):

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape
    crow, ccol = rows//2 , cols//2

    mask = np.ones((rows, cols), np.float32)

    # suppress grid frequencies
    mask[crow-3:crow+3, :] = 0.3
    mask[:, ccol-3:ccol+3] = 0.3

    fshift_filtered = fshift * mask

    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)

    img_back = np.abs(img_back)

    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = img_back.astype(np.uint8)

    return img_back


# ===== GRID DETECTION =====
def detect_grid(gray):

    edges = cv2.Canny(gray, 50, 150)

    horizontal = np.sum(edges, axis=1)
    vertical = np.sum(edges, axis=0)

    h_score = np.std(horizontal)
    v_score = np.std(vertical)

    return (h_score + v_score) > 40


# ===== PROCESS ALL IMAGES =====
for file in os.listdir(INPUT_DIR):

    path = os.path.join(INPUT_DIR, file)
    img = cv2.imread(path)

    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: detect if grid exists
    if detect_grid(gray):
        processed = fft_remove_grid(gray)
    else:
        processed = gray

    # Step 2: threshold
    _, clean = cv2.threshold(
        processed,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Step 3: connect tiny ECG gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,1))
    connected = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    final = np.where(connected > 0, clean | connected, 0).astype(np.uint8)

    # Step 4: skeletonize ECG
    skeleton = skeletonize(final)

    # Step 5: redraw ECG
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    redrawn = cv2.dilate(skeleton, kernel, iterations=1)

    redrawn = cv2.GaussianBlur(redrawn, (3,3), 0)
    _, redrawn = cv2.threshold(redrawn, 127, 255, cv2.THRESH_BINARY)

    # Save
    save_path = os.path.join(OUTPUT_DIR, file)
    cv2.imwrite(save_path, redrawn)

    print(f"Processed: {file}")

print("✅ All ECG images processed successfully.")