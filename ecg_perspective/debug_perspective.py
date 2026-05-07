import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ================= PATHS =================
INPUT_DIR = r"C:/Academics/FOURTH YEAR/FINAL YEAR PROJECT/Project2025_26/ECG_dataset/images"
DEBUG_DIR = r"C:/Academics/FOURTH YEAR/FINAL YEAR PROJECT/Project2025_26/ECG_dataset/debug_perspective"

os.makedirs(DEBUG_DIR, exist_ok=True)

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

def debug_corner_detection(image, filename):
    """
    Debug version that shows each step of corner detection.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Step 1: Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 2: Try different edge detection approaches
    # Approach 1: Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Approach 2: Adaptive threshold
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
    
    # Approach 3: Simple threshold (assumes white paper on dark background)
    _, thresh_simple = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 3: Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Step 4: Find contours on different images
    contours_adaptive, _ = cv2.findContours(binary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_simple, _ = cv2.findContours(thresh_simple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_edges, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create debug visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Debug: {filename}', fontsize=16)
    
    # Row 1: Original processing
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(blurred, cmap='gray')
    axes[0, 2].set_title('Blurred')
    axes[0, 2].axis('off')
    
    # Row 2: Different thresholding methods
    axes[1, 0].imshow(edges, cmap='gray')
    axes[1, 0].set_title(f'Canny Edges ({len(contours_edges)} contours)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(binary, cmap='gray')
    axes[1, 1].set_title(f'Adaptive Threshold ({len(contours_adaptive)} contours)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(thresh_simple, cmap='gray')
    axes[1, 2].set_title(f'Otsu Threshold ({len(contours_simple)} contours)')
    axes[1, 2].axis('off')
    
    # Row 3: Contour detection results
    # Draw all contours on adaptive
    img_contours_adaptive = image.copy()
    if contours_adaptive:
        largest_adaptive = max(contours_adaptive, key=cv2.contourArea)
        cv2.drawContours(img_contours_adaptive, [largest_adaptive], -1, (0, 255, 0), 3)
        area_ratio = cv2.contourArea(largest_adaptive) / (h * w)
        axes[2, 0].set_title(f'Adaptive - Largest Contour\nArea: {area_ratio:.2%} of image')
    else:
        axes[2, 0].set_title('Adaptive - No Contours')
    axes[2, 0].imshow(cv2.cvtColor(img_contours_adaptive, cv2.COLOR_BGR2RGB))
    axes[2, 0].axis('off')
    
    # Draw all contours on simple
    img_contours_simple = image.copy()
    if contours_simple:
        largest_simple = max(contours_simple, key=cv2.contourArea)
        cv2.drawContours(img_contours_simple, [largest_simple], -1, (255, 0, 0), 3)
        area_ratio = cv2.contourArea(largest_simple) / (h * w)
        axes[2, 1].set_title(f'Otsu - Largest Contour\nArea: {area_ratio:.2%} of image')
    else:
        axes[2, 1].set_title('Otsu - No Contours')
    axes[2, 1].imshow(cv2.cvtColor(img_contours_simple, cv2.COLOR_BGR2RGB))
    axes[2, 1].axis('off')
    
    # Try to detect corners and show result
    img_corners = image.copy()
    corners = None
    
    # Try with adaptive threshold contours
    if contours_adaptive:
        largest = max(contours_adaptive, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            for pt in corners:
                cv2.circle(img_corners, tuple(pt.astype(int)), 10, (0, 0, 255), -1)
            axes[2, 2].set_title(f'Detected 4 Corners (Approx)')
        else:
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            corners = box.astype(np.float32)
            for pt in corners:
                cv2.circle(img_corners, tuple(pt.astype(int)), 10, (255, 0, 255), -1)
            axes[2, 2].set_title(f'Detected Corners (MinAreaRect)\nApprox had {len(approx)} points')
    else:
        axes[2, 2].set_title('No Corners Detected')
    
    axes[2, 2].imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    debug_path = os.path.join(DEBUG_DIR, f'debug_{filename}.png')
    plt.savefig(debug_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Debug visualization saved: debug_{filename}.png")
    return corners

# ================= MAIN DEBUG =================

image_files = sorted(
    f for f in os.listdir(INPUT_DIR) 
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
)

print(f"🔍 Debugging {len(image_files)} images...\n")

for fname in image_files[:3]:  # Debug first 3 images
    path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(path)

    if img is None:
        print(f"⚠️ Skipping unreadable: {fname}")
        continue

    print(f"\n{'='*60}")
    print(f"Processing: {fname}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    
    corners = debug_corner_detection(img, fname)
    
    if corners is not None:
        print(f"✅ Corners detected:")
        for i, corner in enumerate(corners):
            print(f"   Corner {i+1}: ({corner[0]:.1f}, {corner[1]:.1f})")
    else:
        print(f"❌ No corners detected")

print(f"\n🎉 Debug complete! Check {DEBUG_DIR} for visualizations.")
