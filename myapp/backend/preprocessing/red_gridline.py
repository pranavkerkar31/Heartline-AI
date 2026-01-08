import cv2
import numpy as np
import matplotlib.pyplot as plt


# -------------------- LOAD IMAGE --------------------
img = cv2.imread("image.png")
if img is None:
    raise FileNotFoundError("ECG image not found")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# -------------------- GRAYSCALE --------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# -------------------- REMOVE GRID TEXTURE (BLACK-HAT) --------------------
# Suppresses repetitive grid background while keeping ECG trace
bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_kernel)

# Normalize to full range
blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)


# -------------------- BINARIZE ECG TRACE --------------------
_, bw = cv2.threshold(
    blackhat,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)


# -------------------- REMOVE GRID LINES (CRITICAL STEP) --------------------
# Detect horizontal grid lines
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)

# Detect vertical grid lines
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

# Combine grid lines
grid = cv2.bitwise_or(horizontal, vertical)

# Subtract grid from binary image
bw = cv2.subtract(bw, grid)


# -------------------- REMOVE SMALL NOISE --------------------
kernel_small = np.ones((2, 2), np.uint8)
bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_small)


# -------------------- CONNECT ECG WAVEFORM --------------------
# Connect broken ECG segments without restoring grid
kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel_connect)


# -------------------- FINAL OUTPUT --------------------
# Black waveform on white background
final = 255 - bw


# -------------------- DISPLAY RESULTS --------------------
plt.figure(figsize=(18, 6))

plt.subplot(1, 4, 1)
plt.imshow(rgb)
plt.title("Original ECG")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(blackhat, cmap="gray")
plt.title("After Black-hat (Grid Suppressed)")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(grid, cmap="gray")
plt.title("Detected Grid Lines")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(final, cmap="gray")
plt.title("ECG Waveform Only (FINAL)")
plt.axis("off")

plt.tight_layout()
plt.show()


# -------------------- SAVE OUTPUT --------------------
cv2.imwrite("ecg_waveform_only.png", final)
