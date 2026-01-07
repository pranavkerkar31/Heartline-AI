import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ---------------- STEP A1: Detect & Crop ECG Paper ----------------
def detect_and_crop_paper(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            return img, approx

    return img, None


# ---------------- STEP A2: Perspective Correction ----------------
def warp_perspective(img, pts):
    pts = pts.reshape(4, 2)

    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return warped


# ---------------- STEP A3: Shadow Normalization ----------------
def remove_shadows(gray):
    dilated = cv2.dilate(gray, np.ones((15, 15), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(gray, bg)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return norm


# ---------------- STEP A4: Contrast Normalization ----------------
def enhance_contrast(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ---------------- STEP A5: Background Whitening ----------------
def make_background_white(gray):
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(norm, (5, 5), 0)
    _, white_bg = cv2.threshold(blur, 200, 255, cv2.THRESH_TRUNC)
    return white_bg


# ---------------- MAIN ----------------
if __name__ == "__main__":

    img = cv2.imread("image.jpeg")
    if img is None:
        raise ValueError("Image not found")

    # Create output folder
    output_dir = "stageA_output"
    os.makedirs(output_dir, exist_ok=True)

    # A1: Detect ECG paper
    _, paper_cnt = detect_and_crop_paper(img)

    # A2: Perspective correction
    if paper_cnt is not None:
        paper = warp_perspective(img, paper_cnt)
    else:
        paper = img

    # Convert to grayscale
    gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)

    # A3: Shadow normalization
    shadow_free = remove_shadows(gray)

    # A4: Contrast enhancement
    contrast_ecg = enhance_contrast(shadow_free)

    # A5: Background whitening
    digital_ecg = make_background_white(contrast_ecg)

    # ---------------- SAVE OUTPUTS ----------------
    cv2.imwrite(os.path.join(output_dir, "paper_extracted.jpg"), paper)
    cv2.imwrite(os.path.join(output_dir, "digital_ecg.png"), digital_ecg)

    print("✅ Stage-A outputs saved in:", output_dir)

    # ---------------- DISPLAY ----------------
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Phone Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Paper Extracted & Warped")
    plt.imshow(cv2.cvtColor(paper, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Stage-A Output (Digital ECG-like)")
    plt.imshow(digital_ecg, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
