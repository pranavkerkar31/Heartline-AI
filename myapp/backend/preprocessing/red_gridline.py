import cv2
import numpy as np

# 1. Read image
img = cv2.imread("image.png")   # change name if needed
if img is None:
    raise ValueError("Could not read image")

# --- optional: crop to ECG area only (avoid white border) ---
# Manually tune these once for your scan size
# y1, y2, x1, x2 = 150, 1850, 200, 2600
# img = img[y1:y2, x1:x2]

# 2. Convert to HSV to isolate bright pink/red grid
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Stricter ranges: BRIGHT, fairly saturated pink/red
lower_red1 = np.array([0,   80, 190], dtype=np.uint8)
upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
lower_red2 = np.array([170, 80, 190], dtype=np.uint8)
upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)

# (debug) save mask to verify only grid is selected
cv2.imwrite("debug_red_mask.png", red_mask)

# 3. Fade grid instead of deleting everything red
no_grid = img.copy()
red_idx = red_mask == 255

# Make grid very light but not pure white
no_grid[red_idx] = cv2.addWeighted(
    no_grid[red_idx], 0.2,                      # keep 20% of original
    np.full_like(no_grid[red_idx], 255), 0.8,   # mix with white
    0
)

cv2.imwrite("step_no_grid_faded.png", no_grid)

# 4. Convert to grayscale
gray = cv2.cvtColor(no_grid, cv2.COLOR_BGR2GRAY)

# 5. Threshold to keep only dark trace
# Use fixed high threshold (tune 200–220 if needed)
_, bw = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)

cv2.imwrite("step_bw_raw.png", bw)

# 6. Morphological cleanup: remove isolated noise, keep thin lines
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
waveform = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

# (optional) small closing to connect broken segments
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
waveform = cv2.morphologyEx(waveform, cv2.MORPH_CLOSE, kernel_close, iterations=1)

# 7. Save final outputs
cv2.imwrite("ecg_no_grid.png", no_grid)             # ECG with grid almost gone
cv2.imwrite("ecg_waveform_mask.png", waveform)      # binary mask of waveform only
