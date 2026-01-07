import cv2
import numpy as np
import matplotlib.pyplot as plt


def remove_patient_info_and_redaction(img,
                                      black_threshold=5,
                                      density_threshold=0.02):
    """
    Removes patient info (text) and black redacted regions
    from the top of the ECG image using row-wise analysis.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Step 1: Remove pure black rows (redacted bars)
    row_mean = np.mean(gray, axis=1)
    non_black_rows = row_mean > black_threshold
    img = img[non_black_rows, :]
    gray = gray[non_black_rows, :]

    # --- Step 2: Remove patient info text region
    # Binarize to highlight text
    _, binary = cv2.threshold(
        gray, 200, 255, cv2.THRESH_BINARY_INV
    )

    # Row-wise pixel density
    row_density = np.sum(binary > 0, axis=1) / binary.shape[1]

    # ECG region starts where density increases significantly
    start_row = 0
    for i, density in enumerate(row_density):
        if density > density_threshold:
            start_row = i
            break

    cleaned_img = img[start_row:, :]

    return cleaned_img


# ---------------- MAIN (Testing) ----------------
if __name__ == "__main__":

    image_path = "image.png"   # your ECG image
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found. Check the path.")

    cleaned_img = remove_patient_info_and_redaction(img)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original ECG")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("After Patient Info Removal")
    plt.imshow(cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()
