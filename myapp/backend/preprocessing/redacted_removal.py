import cv2
import numpy as np
import matplotlib.pyplot as plt


def remove_right_and_bottom_info(
    img,
    min_right_strip_width=200,
    bottom_dark_ratio=0.15,
    bottom_search_ratio=0.25
):
    """
    Removes:
    1. Right-side vertical patient info
    2. Bottom black printed text strip
    """

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ==================================================
    # PART 1: REMOVE RIGHT-SIDE PATIENT INFO
    # ==================================================
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.abs(sobelx)
    sobelx = np.uint8(255 * sobelx / (np.max(sobelx) + 1e-6))

    _, binary_x = cv2.threshold(sobelx, 50, 255, cv2.THRESH_BINARY)

    col_density = np.sum(binary_x > 0, axis=0)
    col_density = col_density / (np.max(col_density) + 1e-6)

    cutoff_col = w
    for i in range(w - 1, 0, -1):
        if col_density[i] > 0.35:
            cutoff_col = i
        else:
            break

    cutoff_col = max(cutoff_col - min_right_strip_width, 0)
    img = img[:, :cutoff_col]
    gray = gray[:, :cutoff_col]

    # ==================================================
    # PART 2: REMOVE BOTTOM BLACK TEXT STRIP
    # ==================================================
    h2 = gray.shape[0]

    # Threshold dark pixels
    _, binary_dark = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

    row_dark_ratio = np.sum(binary_dark > 0, axis=1) / gray.shape[1]

    # Only scan bottom portion
    start_search = int(h2 * (1 - bottom_search_ratio))

    cutoff_row = h2
    for i in range(h2 - 1, start_search, -1):
        if row_dark_ratio[i] > bottom_dark_ratio:
            cutoff_row = i
        else:
            break

    cleaned_img = img[:cutoff_row, :]

    return cleaned_img


# ---------------- TEST ----------------
if __name__ == "__main__":

    img = cv2.imread("trial.png")
    if img is None:
        raise ValueError("Image not found")

    cleaned = remove_right_and_bottom_info(img)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Right + Bottom Info Removed")
    plt.imshow(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()
