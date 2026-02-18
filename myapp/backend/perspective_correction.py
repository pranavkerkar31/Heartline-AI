import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ================= PATHS =================
INPUT_DIR = "ECG_dataset/images"
OUTPUT_DIR = "perspective_corrected"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= HOMOGRAPHY-BASED PERSPECTIVE CORRECTION =================
# This implementation uses:
# 1. Document corner detection via contour analysis
# 2. Homography matrix computation using cv2.findHomography() with RANSAC
# 3. Perspective warping using cv2.warpPerspective() with high-quality interpolation
# ============================================================================

def order_points(pts):
    """
    Order points in top-left, top-right, bottom-right, bottom-left order.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum of coordinates: top-left (min sum), bottom-right (max sum)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Difference of coordinates: top-right (min diff), bottom-left (max diff)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def detect_document_corners(image):
    """
    Detect the 4 corners of the document using corner detection.
    Returns the 4 corner points or None if detection fails.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Apply adaptive thresholding to get binary image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour (assumed to be the document)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the convex hull and approximate to polygon
    hull = cv2.convexHull(largest_contour)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
    
    # If we have 4 points, use them directly
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)
    
    # Otherwise, get the minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    
    # Filter out rectangles that are too small (likely not the document)
    area = cv2.contourArea(box)
    image_area = h * w
    if area < 0.1 * image_area:  # Document should be at least 10% of image
        return None
    
    return box.astype(np.float32)

def apply_homography_transform(image, src_points):
    """
    Apply homography transformation using detected corners.
    Uses cv2.findHomography with RANSAC for robust estimation.
    """
    # Order the source points
    src = order_points(src_points)
    (tl, tr, br, bl) = src
    
    # Compute the width and height of the destination image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Define destination points (perfect rectangle)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype=np.float32)
    
    # Compute homography matrix using RANSAC for robustness
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    
    if H is None:
        return None
    
    # Apply perspective warp using the homography matrix
    warped = cv2.warpPerspective(image, H, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)
    
    return warped

def correct_perspective(image):
    """
    Correct perspective using Homography + Perspective Warp.
    """
    # 1. Detect the 4 corners of the document
    corners = detect_document_corners(image)
    
    if corners is None:
        return image, False
    
    # 2. Apply homography transformation
    warped = apply_homography_transform(image, corners)
    
    if warped is None:
        return image, False
    
    return warped, True

# ================= MAIN PROCESSING =================

image_files = sorted(
    f for f in os.listdir(INPUT_DIR) 
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
)

print(f"📂 Found {len(image_files)} images for homography-based perspective correction...")

success_count = 0
failed_images = []

for fname in image_files:
    path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(path)

    if img is None:
        print(f"⚠️ Skipping unreadable: {fname}")
        continue

    # Apply homography-based correction
    result, corrected = correct_perspective(img)
    
    # Save output with high quality JPEG setting
    out_path = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(out_path, result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    if corrected:
        success_count += 1
        print(f"✅ Corrected: {fname}")
    else:
        failed_images.append(fname)
        print(f"⚠️ No contour found (kept original): {fname}")

# ================= SUCCESS RATE CALCULATION =================
total_images = len(image_files)
success_percentage = (success_count / total_images * 100) if total_images > 0 else 0
failure_count = total_images - success_count

print(f"\n🎉 Homography-based perspective correction complete.")
print(f"\n📊 CORRECTION STATISTICS:")
print(f"   Total Images Processed: {total_images}")
print(f"   Successfully Corrected: {success_count}")
print(f"   Failed Corrections: {failure_count}")
print(f"   Success Rate: {success_count}/{total_images} ({success_percentage:.2f}%)")

# Save detailed report to file
report_path = os.path.join(OUTPUT_DIR, "correction_report.txt")
with open(report_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("PERSPECTIVE CORRECTION REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total Images Processed: {total_images}\n")
    f.write(f"Successfully Corrected: {success_count}\n")
    f.write(f"Failed Corrections: {failure_count}\n")
    f.write(f"Success Rate: {success_percentage:.2f}%\n\n")
    
    if failed_images:
        f.write("=" * 60 + "\n")
        f.write("FAILED IMAGES LIST:\n")
        f.write("=" * 60 + "\n")
        for i, fname in enumerate(failed_images, 1):
            f.write(f"{i}. {fname}\n")
    else:
        f.write("All images were successfully corrected!\n")

print(f"\n📄 Detailed report saved to: {report_path}")
