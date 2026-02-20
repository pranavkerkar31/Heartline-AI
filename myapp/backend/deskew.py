import cv2
import numpy as np
import os

# =====================================================
# Method 1: Hough-based deskew
# =====================================================
def deskew_hough(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return None, None

    angles = []
    for i in range(min(50, len(lines))):
        rho, theta = lines[i][0]
        angle = (theta - np.pi / 2) * 180 / np.pi
        if -10 < angle < 10:
            angles.append(angle)

    if len(angles) < 5:
        return None, None

    median_angle = np.median(angles)

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated, median_angle


# =====================================================
# Method 2: Projection-profile deskew (fallback)
# =====================================================
def deskew_projection(img, angle_range=5, step=0.2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    h, w = bw.shape
    center = (w // 2, h // 2)

    best_angle = 0
    best_score = -1

    for angle in np.arange(-angle_range, angle_range + step, step):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            bw, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        projection = np.sum(rotated, axis=1)
        score = np.var(projection)

        if score > best_score:
            best_score = score
            best_angle = angle

    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return corrected, best_angle


# =====================================================
# Unified deskew function (AUTO SELECT)
# =====================================================
def deskew_ecg(img):
    # Try Hough first
    hough_img, hough_angle = deskew_hough(img)

    if hough_img is not None and abs(hough_angle) <= 5:
        print(f"✔ Hough used | angle = {hough_angle:.2f}°")
        return hough_img, hough_angle

    # Fallback to projection method
    proj_img, proj_angle = deskew_projection(img)
    print(f"✔ Projection used | angle = {proj_angle:.2f}°")

    return proj_img, proj_angle


# =====================================================
# Run on folder or single image
# =====================================================
def process_path(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    if os.path.isfile(input_path):
        img = cv2.imread(input_path)
        out, angle = deskew_ecg(img)
        cv2.imwrite(os.path.join(output_path, os.path.basename(input_path)), out)

    else:
        for file in os.listdir(input_path):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(input_path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            out, angle = deskew_ecg(img)
            cv2.imwrite(os.path.join(output_path, file), out)


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    INPUT_PATH = "WhatsApp Image 2026-02-16 at 1.04.40 PM.jpeg"      # folder or single image
    OUTPUT_PATH = "deskewed_ecg"

    process_path(INPUT_PATH, OUTPUT_PATH)
