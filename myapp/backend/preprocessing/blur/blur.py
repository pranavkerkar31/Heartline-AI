import cv2
import numpy as np
import os

# --------------------------------------------------
# Convert to grayscale
# --------------------------------------------------
def to_grayscale(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# --------------------------------------------------
# Blur detection: Tenengrad (ECG-safe)
# --------------------------------------------------
def tenengrad_score(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(gx**2 + gy**2)


# --------------------------------------------------
# Very mild enhancement (NOT deblurring)
# --------------------------------------------------
def mild_enhancement(gray):
    """
    Only improves edge continuity slightly.
    Does NOT attempt deblurring.
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    enhanced = cv2.addWeighted(gray, 1.2, blur, -0.2, 0)
    return enhanced


# --------------------------------------------------
# Kernel-11 ONLY blur handling
# --------------------------------------------------
def blur_handling_kernel11(input_img):
    gray = to_grayscale(input_img)
    score = tenengrad_score(gray)

    # ---- Kernel-11 acceptance threshold ----
    # Empirically safe for ECG images
    if score > 150:
        output = mild_enhancement(gray)
        status = "accepted_kernel_11"
    else:
        output = None
        status = "rejected_not_kernel_11"

    return output, score, status


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":

    # 🔹 Input image (kernel-11 ECG)
    img_path = "kernel11.jpeg"

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"{img_path} not found")

    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("Unable to read image")

    out, score, status = blur_handling_kernel11(img)

    print("\n--- BLUR ANALYSIS REPORT (KERNEL-11 ONLY) ---")
    print(f"Input image     : {img_path}")
    print(f"Tenengrad score : {score:.2f}")
    print(f"Status          : {status}")

    if out is not None:
        output_path = "kernel11_processed.png"
        cv2.imwrite(output_path, out)
        print(f"Output saved    : {output_path}")
    else:
        print("Image rejected (not kernel-11 quality)")
