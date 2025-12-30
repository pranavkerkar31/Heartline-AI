import cv2
import numpy as np
import matplotlib.pyplot as plt


def remove_redacted_area(img, intensity_threshold=5):
    """
    Removes black/redacted regions using row-wise average pixel intensity.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row_mean_intensity = np.mean(gray, axis=1)

    valid_rows = row_mean_intensity > intensity_threshold
    cropped_img = img[valid_rows, :]

    return cropped_img


def remove_gridlines_and_background(img, threshold=0.94):
    """
    Removes ECG gridlines and background by extracting the red channel,
    converting to grayscale, and applying intensity thresholding.
    """

    # Extract red channel (OpenCV uses BGR)
    red_channel = img[:, :, 2]

    # Normalize to [0, 1]
    red_norm = red_channel / 255.0

    # Thresholding
    _, binary = cv2.threshold(
        red_norm, threshold, 1.0, cv2.THRESH_BINARY_INV
    )

    # Convert to uint8 image
    binary = (binary * 255).astype(np.uint8)

    return binary

if __name__ == "__main__":

    image_path = "image.png"   # update if needed
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found. Check the path.")

    # Step 1.2: Remove redacted area
    img_no_redaction = remove_redacted_area(img)

    # Step 1.3: Remove gridlines & background
    binary_ecg = remove_gridlines_and_background(img_no_redaction)

    # Display results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original ECG")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("After Redacted Area Removal")
    plt.imshow(cv2.cvtColor(img_no_redaction, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Binary ECG (Gridlines Removed)")
    plt.imshow(binary_ecg, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
