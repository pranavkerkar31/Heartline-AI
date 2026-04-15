import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_DIR = r"C:/Academics/FOURTH YEAR/FINAL YEAR PROJECT/Project2025_26/ECG_dataset/images"
OUTPUT_DIR = r"C:/Academics/FOURTH YEAR/FINAL YEAR PROJECT/Project2025_26/ECG_dataset/corrected_perspective"
VERIFY_DIR = r"C:/Academics/FOURTH YEAR/FINAL YEAR PROJECT/Project2025_26/ECG_dataset/verification"

os.makedirs(VERIFY_DIR, exist_ok=True)

def order_points(pts):
    """Order points in top-left, top-right, bottom-right, bottom-left order."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def extract_corners(contour):
    """Extract 4 corners from a contour."""
    if contour is None or len(contour) < 4:
        return None, None
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for epsilon_factor in [0.02, 0.03, 0.04, 0.05, 0.06]:
        approx = cv2.approxPolyDP(hull, epsilon_factor * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32), "approx"
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32), "minAreaRect"

def score_corner_detection(corners, img_w, img_h, area_ratio, corner_method):
    """Score the quality of corner detection."""
    if corners is None:
        return 0
    ordered = order_points(corners)
    score = 0
    
    if corner_method == "approx":
        score += 150
    else:
        score += 50
    
    angles = []
    for i in range(4):
        p1 = ordered[i]
        p2 = ordered[(i+1)%4]
        p3 = ordered[(i+2)%4]
        v1 = p1 - p2
        v2 = p3 - p2
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
        angles.append(np.degrees(angle))
    
    avg_angle_deviation = np.mean([abs(angle - 90) for angle in angles])
    if avg_angle_deviation < 15:
        score += 120
    elif avg_angle_deviation < 25:
        score += 90
    elif avg_angle_deviation < 40:
        score += 50
    else:
        score += 10
    
    corner_spread_x = np.max(corners[:, 0]) - np.min(corners[:, 0])
    corner_spread_y = np.max(corners[:, 1]) - np.min(corners[:, 1])
    spread_ratio_x = corner_spread_x / img_w
    spread_ratio_y = corner_spread_y / img_h
    
    if spread_ratio_x > 0.6 and spread_ratio_y > 0.5:
        score += 80
    elif spread_ratio_x > 0.4 and spread_ratio_y > 0.3:
        score += 50
    elif spread_ratio_x > 0.25 and spread_ratio_y > 0.2:
        score += 20
    
    if 0.3 <= area_ratio <= 0.75:
        score += 40
    elif 0.15 <= area_ratio <= 0.85:
        score += 30
    elif 0.03 <= area_ratio <= 0.95:
        score += 20
    else:
        score += 5
    
    margin = 20
    at_border = sum(1 for corner in corners for x, y in [corner] 
                   if x < margin or x > img_w - margin or y < margin or y > img_h - margin)
    
    if at_border == 4:
        score -= 150
    elif at_border <= 2:
        score += 30
    
    return score

def detect_document_corners(image):
    """Multi-method document corner detection."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_area = h * w
    
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    binary_otsu = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary_otsu = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, kernel, iterations=2)
    
    edges = cv2.Canny(blurred, 30, 150)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    best_corners = None
    best_score = 0
    best_method = None
    
    for method_name, binary_img in [('Otsu', binary_otsu), ('Canny', edges_dilated)]:
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            area = cv2.contourArea(cnt)
            area_ratio = area / image_area
            if not (0.02 < area_ratio < 0.98):
                continue
            
            corners, corner_method = extract_corners(cnt)
            if corners is None:
                continue
            
            score = score_corner_detection(corners, w, h, area_ratio, corner_method)
            if score > best_score:
                best_score = score
                best_corners = corners
                best_method = f"{method_name}-{corner_method}"
    
    return best_corners, best_method, best_score

# Verify all images
image_files = ['test1.jpeg', 'test2.jpeg', 'test3.jpeg']

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Perspective Correction Verification', fontsize=16, fontweight='bold')

for idx, fname in enumerate(image_files):
    original_path = os.path.join(INPUT_DIR, fname)
    corrected_path = os.path.join(OUTPUT_DIR, fname)
    
    original = cv2.imread(original_path)
    corrected = cv2.imread(corrected_path)
    
    # Detect corners
    corners, method, score = detect_document_corners(original)
    
    # Draw corners on original
    img_with_corners = original.copy()
    if corners is not None:
        ordered = order_points(corners)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        labels = ['TL', 'TR', 'BR', 'BL']
        
        for i, (pt, color, label) in enumerate(zip(ordered, colors, labels)):
            pt_int = tuple(pt.astype(int))
            cv2.circle(img_with_corners, pt_int, 15, color, -1)
            cv2.circle(img_with_corners, pt_int, 17, (255, 255, 255), 2)
            cv2.putText(img_with_corners, label, (pt_int[0]-25, pt_int[1]-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        # Draw lines
        for i in range(4):
            pt1 = tuple(ordered[i].astype(int))
            pt2 = tuple(ordered[(i+1)%4].astype(int))
            cv2.line(img_with_corners, pt1, pt2, (255, 0, 255), 3)
    
    # Plot
    axes[idx, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[idx, 0].set_title(f'{fname}\nOriginal', fontsize=10)
    axes[idx, 0].axis('off')
    
    axes[idx, 1].imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
    title = f'Detected Corners\n{method} (score: {score:.0f})' if corners is not None else 'No Corners Detected'
    axes[idx, 1].set_title(title, fontsize=10)
    axes[idx, 1].axis('off')
    
    axes[idx, 2].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    axes[idx, 2].set_title(f'Corrected Result\n{corrected.shape[1]}x{corrected.shape[0]}', fontsize=10)
    axes[idx, 2].axis('off')
    
    print(f"✅ {fname}")
    print(f"   Method: {method} | Score: {score:.0f}")
    if corners is not None:
        ordered = order_points(corners)
        print(f"   TL: ({ordered[0][0]:.0f}, {ordered[0][1]:.0f}) | TR: ({ordered[1][0]:.0f}, {ordered[1][1]:.0f})")
        print(f"   BR: ({ordered[2][0]:.0f}, {ordered[2][1]:.0f}) | BL: ({ordered[3][0]:.0f}, {ordered[3][1]:.0f})")
    print()

plt.tight_layout()
verify_path = os.path.join(VERIFY_DIR, 'verification_all_images.png')
plt.savefig(verify_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"🎉 Verification complete!")
print(f"📁 Saved to: {verify_path}")
