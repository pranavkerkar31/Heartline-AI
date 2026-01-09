import cv2
import numpy as np

# Load ECG image
image = cv2.imread("ecg.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ---- ROW-WISE ANALYSIS ----
row_mean = gray.mean(axis=1)
low_thresh, high_thresh = 15, 240
valid_rows = np.where((row_mean > low_thresh) & (row_mean < high_thresh))[0]
top, bottom = valid_rows[0], valid_rows[-1]

# ---- COLUMN-WISE ANALYSIS ----
col_mean = gray.mean(axis=0)
valid_cols = np.where((col_mean > low_thresh) & (col_mean < high_thresh))[0]
left, right = valid_cols[0], valid_cols[-1]

# ---- CROP ECG REGION ----
ecg_only = image[top:bottom, left:right]

# Save result
cv2.imwrite("ecg_redacted.jpg", ecg_only)
