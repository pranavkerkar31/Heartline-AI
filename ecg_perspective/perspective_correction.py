import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ================= PATHS =================
INPUT_DIR = r"C:/Academics/FOURTH YEAR/FINAL YEAR PROJECT/Project2025_26/ECG_dataset/augmented_agcwd"
OUTPUT_DIR = r"C:/Academics/FOURTH YEAR/FINAL YEAR PROJECT/Project2025_26/ECG_dataset/corrected_perspective"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= HOMOGRAPHY-BASED PERSPECTIVE CORRECTION =================
# This implementation uses:
# 1. Otsu's thresholding for robust paper/background segmentation
# 2. Morphological operations to clean paper boundaries
# 3. Contour area and shape filtering to identify the document
# 4. Homography matrix computation using cv2.findHomography() with RANSAC
# 5. Perspective warping using cv2.warpPerspective() with high-quality interpolation
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
    Multi-method document corner detection with fallback strategies:
    1. Otsu's thresholding for paper/background separation
    2. Canny edge detection for tilted papers
    3. Contour filtering by area and shape
    4. Handles papers that extend beyond image boundaries
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_area = h * w
    
    # Apply blur to remove text/grid details
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Method 1: Otsu's thresholding (works well for clear paper/background separation)
    _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    binary_otsu = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary_otsu = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Method 2: Canny edge detection (better for extreme angles/cropped papers)
    edges = cv2.Canny(blurred, 30, 150)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Try both methods and pick the best result
    best_corners = None
    best_score = 0
    best_method = None
    
    for method_name, binary_img in [('Otsu', binary_otsu), ('Canny', edges_dilated)]:
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        # Try each significant contour
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            area = cv2.contourArea(cnt)
            area_ratio = area / image_area
            
            # Document should be between 2% and 98% of image area
            # (Low threshold allows edge-only detections like Canny)
            if not (0.02 < area_ratio < 0.98):
                continue
            
            # Extract corners
            corners, corner_method = extract_corners(cnt)
            
            if corners is None:
                continue
            
            # Score this detection
            score = score_corner_detection(corners, w, h, area_ratio, corner_method)
            
            if score > best_score:
                best_score = score
                best_corners = corners
                best_method = f"{method_name}-{corner_method}"
    
    return best_corners

def score_corner_detection(corners, img_w, img_h, area_ratio, corner_method):
    """
    Score the quality of corner detection.
    Higher score = better detection.
    Prioritizes corner quality over contour area.
    """
    if corners is None:
        return 0
    
    # Order corners
    ordered = order_points(corners)
    
    # Calculate various quality metrics
    score = 0
    
    # 1. Prefer 'approx' method (means 4 clean corners found) over minAreaRect
    if corner_method == "approx":
        score += 150  # Strong bonus for clean 4-corner detection
    else:
        score += 50   # minAreaRect is fallback
    
    # 2. Check if corners form a reasonable quadrilateral
    # Calculate angles at each corner
    angles = []
    for i in range(4):
        p1 = ordered[i]
        p2 = ordered[(i+1)%4]
        p3 = ordered[(i+2)%4]
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
        angles.append(np.degrees(angle))
    
    # Good rectangles have angles close to 90 degrees (MOST IMPORTANT)
    avg_angle_deviation = np.mean([abs(angle - 90) for angle in angles])
    if avg_angle_deviation < 15:
        score += 120  # Excellent rectangle
    elif avg_angle_deviation < 25:
        score += 90   # Good rectangle
    elif avg_angle_deviation < 40:
        score += 50   # Acceptable
    else:
        score += 10   # Poor geometry
    
    # 3. Check if corners are well-distributed (not all clustered)
    corner_spread_x = np.max(corners[:, 0]) - np.min(corners[:, 0])
    corner_spread_y = np.max(corners[:, 1]) - np.min(corners[:, 1])
    
    spread_ratio_x = corner_spread_x / img_w
    spread_ratio_y = corner_spread_y / img_h
    
    if spread_ratio_x > 0.6 and spread_ratio_y > 0.5:
        score += 80   # Excellent spread
    elif spread_ratio_x > 0.4 and spread_ratio_y > 0.3:
        score += 50   # Good spread
    elif spread_ratio_x > 0.25 and spread_ratio_y > 0.2:
        score += 20   # Acceptable
    
    # 4. Area coverage (less important now, but still a factor)
    # Paper boundary contours can be small (3-10%), full paper can be 40-70%
    if 0.3 <= area_ratio <= 0.75:
        score += 40   # Ideal range
    elif 0.15 <= area_ratio <= 0.85:
        score += 30   # Good range
    elif 0.03 <= area_ratio <= 0.95:
        score += 20   # Acceptable (includes edge-only detection)
    else:
        score += 5    # Too small or too large
    
    # 5. Penalize if ALL corners are at image borders (likely wrong detection)
    margin = 20
    at_border = 0
    for corner in corners:
        x, y = corner
        if x < margin or x > img_w - margin or y < margin or y > img_h - margin:
            at_border += 1
    
    if at_border == 4:
        score -= 150  # All corners at border = bad
    elif at_border <= 2:
        score += 30   # Some corners at border is OK (cropped paper)
    
    return score

def extract_corners(contour):
    """Extract 4 corners from a contour."""
    if contour is None or len(contour) < 4:
        return None, None
    
    # Get convex hull
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    
    # Approximate to polygon - try different epsilon values to get 4 points
    for epsilon_factor in [0.02, 0.03, 0.04, 0.05, 0.06]:
        approx = cv2.approxPolyDP(hull, epsilon_factor * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32), "approx"
    
    # If we can't get exactly 4 points, use minimum area rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32), "minAreaRect"

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
    Uses multi-method detection with quality scoring.
    """
    # 1. Detect the 4 corners of the document
    corners = detect_document_corners(image)
    
    if corners is None:
        return image, False
    
    # 2. Apply homography transformation
    warped = apply_homography_transform(image, corners)
    
    if warped is None:
        return image, False
    
    # 3. Validate the result (check if warped image is reasonable)
    h_orig, w_orig = image.shape[:2]
    h_warp, w_warp = warped.shape[:2]
    
    # If warped image is nearly identical in size, correction might have failed
    size_change = abs(h_warp - h_orig) + abs(w_warp - w_orig)
    if size_change < 10:  # Less than 10 pixels difference total
        # Check if image actually changed
        if h_warp == h_orig and w_warp == w_orig:
            diff = cv2.absdiff(image, warped)
            if np.sum(diff) < 1000:
                return image, False
    
    return warped, True

# ================= MAIN PROCESSING =================

image_files = sorted(
    f for f in os.listdir(INPUT_DIR) 
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
)

print(f"📂 Found {len(image_files)} images for homography-based perspective correction...")
print()

success_count = 0
failed_images = []

for fname in image_files:
    path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(path)

    if img is None:
        print(f"⚠️ Skipping unreadable: {fname}")
        continue

    print(f"Processing: {fname}...", end=" ")

    # Apply homography-based correction
    result, corrected = correct_perspective(img)
    
    # Save output with high quality JPEG setting
    out_path = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(out_path, result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    if corrected:
        success_count += 1
        print("✅ Corrected")
    else:
        failed_images.append(fname)
        print("⚠️ No correction (kept original)")

# ================= SUCCESS RATE CALCULATION =================
total_images = len(image_files)
success_percentage = (success_count / total_images * 100) if total_images > 0 else 0
failure_count = total_images - success_count

print(f"\n{'='*60}")
print(f"🎉 Homography-based perspective correction complete!")
print(f"\n📊 CORRECTION STATISTICS:")
print(f"   Total Images Processed: {total_images}")
print(f"   Successfully Corrected: {success_count}")
print(f"   Failed Corrections: {failure_count}")
print(f"   Success Rate: {success_count}/{total_images} ({success_percentage:.2f}%)")

# Save detailed report to file
report_path = os.path.join(OUTPUT_DIR, "correction_report.txt")
with open(report_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("PERSPECTIVE CORRECTION REPORT (IMPROVED)\n")
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
