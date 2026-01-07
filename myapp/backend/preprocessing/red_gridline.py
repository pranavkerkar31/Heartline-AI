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

def remove_gridlines_and_background(img):
    """
    Gridline removal adapted for white-background ECGs with pink/red grids.
    Keeps black ECG signal and lead names visible.
    """
    # Extract red channel (grid is prominent here)
    red = img[:, :, 2]
    red_norm = red / 255.0

    # Stronger blur to smooth out fine grid texture while preserving thicker black lines/text
    red_blur = cv2.GaussianBlur(red_norm, (9, 9), 0)

    # Keep mid-range values where gridlines typically sit (pink/red on white ≈ 0.6–0.95 in red channel)
    # Adjust these if needed for your specific scans
    grid_mask = (red_blur > 0.55) & (red_blur < 0.96)

    # Invert: black signal/text becomes 255, background and grid become 0
    binary = np.where(grid_mask, 0, 255).astype(np.uint8)

    # Optional: morphological closing to fill small gaps in traces/text
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary

if __name__ == "__main__":
    image_path = "image.png"  # Replace with your second image filename
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found. Check the path.")

    # Step 1: Remove redacted area (if any black bars at top/bottom)
    img_no_redaction = remove_redacted_area(img)

    # Step 2: Remove gridlines & background
    binary_ecg = remove_gridlines_and_background(img_no_redaction)

    # Display results
    plt.figure(figsize=(15, 5))

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