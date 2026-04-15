import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("MI(95).jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Remove vertical lines (lead separators)
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
remove_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
binary = cv2.subtract(binary, remove_vertical)

# Remove horizontal text (lead labels)
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
remove_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
binary = cv2.subtract(binary, remove_horizontal)

# Thin the waveform (skeletonization)
binary = cv2.ximgproc.thinning(binary)

h, w = binary.shape
signal = []

# Extract waveform
for x in range(w):
    y_pixels = np.where(binary[:, x] == 255)[0]
    if len(y_pixels) > 0:
        y = np.mean(y_pixels)
        signal.append(y)
    else:
        signal.append(signal[-1] if len(signal) > 0 else 0)

signal = np.array(signal)

# Normalize
signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

# Plot
plt.figure(figsize=(12,4))
plt.plot(signal)
plt.title("Extracted ECG Signal (Improved)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()