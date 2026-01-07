import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ---------------- STEP A1: Detect ECG Paper ----------------
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
            return approx

    return None


# ---------------- STEP A2: Perspective Correction ----------------
def warp_perspective(img, pts):
    pts = pts.reshape(4, 2)

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    dst = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (width, height))


# ---------------- STEP A3: Shadow Normalization ----------------
def remove_shadows(gray):
    dilated = cv2.dilate(gray, np.ones((15,15), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(gray, bg)
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)


# ---------------- STEP A4: Contrast Enhancement ----------------
def enhance_contrast(gray):
    return cv2.createCLAHE(2.0, (8,8)).apply(gray)


# ---------------- STEP A5: Background Whitening ----------------
def make_background_white(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, white = cv2.threshold(blur, 200, 255, cv2.THRESH_TRUNC)
    return white


# ---------------- STEP A6: Extract ECG Waveform Area ----------------
def extract_ecg_area(gray):
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150)

    row_density = np.sum(edges > 0, axis=1)
    thresh = 0.12 * np.max(row_density)
    rows = np.where(row_density > thresh)[0]

    valid_rows = []
    for r in rows:
        cols = np.where(edges[r] > 0)[0]
        if len(cols) > 0 and (cols[-1] - cols[0]) > 0.6 * w:
            valid_rows.append(r)

    if len(valid_rows) == 0:
        return gray

    top, bottom = valid_rows[0], valid_rows[-1]
    pad = int(0.01 * h)
    cropped = gray[max(0, top-pad):min(h, bottom+pad), :]

    edges_crop = cv2.Canny(cropped, 50, 150)
    col_density = np.sum(edges_crop > 0, axis=0)
    cols = np.where(col_density > 0.08 * np.max(col_density))[0]

    if len(cols) > 0:
        cropped = cropped[:, cols[0]:cols[-1]]

    return cropped


# ---------------- STEP B: Gridline Removal (Red + Black) ----------------
def remove_gridlines_and_background(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, int(gray.shape[0]*0.05))
    )
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (int(gray.shape[1]*0.05), 1)
    )

    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    grid = cv2.bitwise_or(vertical, horizontal)
    ecg = cv2.subtract(binary, grid)
    ecg = cv2.bitwise_not(ecg)

    ecg = cv2.morphologyEx(ecg, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
    return ecg


# ---------------- MAIN PIPELINE ----------------
if __name__ == "__main__":

    img = cv2.imread("image.jpeg")
    if img is None:
        raise ValueError("Image not found")

    os.makedirs("output", exist_ok=True)

    paper_cnt = detect_and_crop_paper(img)
    paper = warp_perspective(img, paper_cnt) if paper_cnt is not None else img

    gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    shadow_free = remove_shadows(gray)
    contrast = enhance_contrast(shadow_free)
    white_bg = make_background_white(contrast)

    ecg_cropped = extract_ecg_area(white_bg)
    ecg_no_grid = remove_gridlines_and_background(
        cv2.cvtColor(ecg_cropped, cv2.COLOR_GRAY2BGR)
    )

    cv2.imwrite("output/ecg_final_binary.png", ecg_no_grid)

    # -------- DISPLAY --------
    plt.figure(figsize=(15,4))

    plt.subplot(1,4,1)
    plt.title("Paper Extracted")
    plt.imshow(paper, cmap="gray")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.title("Digital ECG (Full)")
    plt.imshow(white_bg, cmap="gray")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.title("ECG Area Only")
    plt.imshow(ecg_cropped, cmap="gray")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.title("Final ECG (No Grid)")
    plt.imshow(ecg_no_grid, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
