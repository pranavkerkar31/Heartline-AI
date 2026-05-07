import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_DIR = r"C:/Academics/FOURTH YEAR/FINAL YEAR PROJECT/Project2025_26/ECG_dataset/images"
DEBUG_DIR = r"C:/Academics/FOURTH YEAR/FINAL YEAR PROJECT/Project2025_26/ECG_dataset/debug_corners"

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

def extract_corners(contour):
    """Extract 4 corners from a contour."""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    
    for epsilon_factor in [0.02, 0.03, 0.04, 0.05]:
        approx = cv2.approxPolyDP(hull, epsilon_factor * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32), "approx", len(approx)
    
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32), "minAreaRect", 4

def detect_document_corners_debug(image, fname):
    """Debug version with multiple threshold methods."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Current approach (Otsu)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    binary_otsu_clean = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary_otsu_clean = cv2.morphologyEx(binary_otsu_clean, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Method 2: Adaptive threshold
    binary_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 21, 10)
    binary_adaptive_clean = cv2.morphologyEx(binary_adaptive, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Method 3: Canny edges
    edges = cv2.Canny(blurred, 30, 150)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Method 4: Two-pass threshold (more aggressive)
    blurred_strong = cv2.GaussianBlur(gray, (21, 21), 0)
    _, binary_strong = cv2.threshold(blurred_strong, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_large = np.ones((15, 15), np.uint8)
    binary_strong_clean = cv2.morphologyEx(binary_strong, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    
    # Find contours for each method
    methods = {
        'Otsu (Current)': binary_otsu_clean,
        'Adaptive': binary_adaptive_clean,
        'Canny Edges': edges_dilated,
        'Strong Otsu': binary_strong_clean
    }
    
    results = {}
    image_area = h * w
    
    for method_name, binary_img in methods.items():
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            results[method_name] = None
            continue
        
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        area_ratio = area / image_area
        
        corners, corner_method, num_points = extract_corners(largest)
        
        results[method_name] = {
            'binary': binary_img,
            'contour': largest,
            'corners': corners,
            'area_ratio': area_ratio,
            'corner_method': corner_method,
            'num_contours': len(contours)
        }
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    fig.suptitle(f'Corner Detection Analysis: {fname}\nImage size: {w}x{h}', 
                 fontsize=16, fontweight='bold')
    
    # Row 1: Binary images
    for idx, (method_name, result) in enumerate(results.items()):
        ax = fig.add_subplot(gs[0, idx])
        if result is not None:
            ax.imshow(result['binary'], cmap='gray')
            ax.set_title(f'{method_name}\nArea: {result["area_ratio"]:.1%} | Contours: {result["num_contours"]}',
                        fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No contours', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(method_name, fontsize=10)
        ax.axis('off')
    
    # Row 2: Detected contours
    for idx, (method_name, result) in enumerate(results.items()):
        ax = fig.add_subplot(gs[1, idx])
        img_contour = image.copy()
        if result is not None:
            cv2.drawContours(img_contour, [result['contour']], -1, (0, 255, 0), 3)
            ax.set_title(f'Detected Contour', fontsize=10)
        else:
            ax.set_title('No Detection', fontsize=10)
        ax.imshow(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
        ax.axis('off')
    
    # Row 3: Corner points
    for idx, (method_name, result) in enumerate(results.items()):
        ax = fig.add_subplot(gs[2, idx])
        img_corners = image.copy()
        if result is not None:
            corners = result['corners']
            ordered = order_points(corners)
            
            # Draw corners with numbers
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            labels = ['TL', 'TR', 'BR', 'BL']
            for i, (pt, color, label) in enumerate(zip(ordered, colors, labels)):
                pt_int = tuple(pt.astype(int))
                cv2.circle(img_corners, pt_int, 15, color, -1)
                cv2.putText(img_corners, label, (pt_int[0]-20, pt_int[1]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw lines connecting corners
            for i in range(4):
                pt1 = tuple(ordered[i].astype(int))
                pt2 = tuple(ordered[(i+1)%4].astype(int))
                cv2.line(img_corners, pt1, pt2, (255, 0, 255), 2)
            
            ax.set_title(f'Corners ({result["corner_method"]})', fontsize=10)
        else:
            ax.set_title('No Corners', fontsize=10)
        ax.imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
        ax.axis('off')
    
    debug_path = os.path.join(DEBUG_DIR, f'corners_analysis_{fname}.png')
    plt.savefig(debug_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*70}")
    print(f"📊 {fname}")
    print(f"{'='*70}")
    for method_name, result in results.items():
        if result:
            print(f"✅ {method_name:20} | Area: {result['area_ratio']:6.1%} | "
                  f"Method: {result['corner_method']:12} | Contours: {result['num_contours']}")
            corners = result['corners']
            print(f"   Corners: TL={corners[0]}, TR={corners[1]}, BR={corners[2]}, BL={corners[3]}")
        else:
            print(f"❌ {method_name:20} | No detection")
    
    print(f"\n📁 Saved: corners_analysis_{fname}.png")
    
    return results

# Process all test images
image_files = ['test1.jpeg', 'test2.jpeg', 'test3.jpeg']

for fname in image_files:
    path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(path)
    
    if img is None:
        print(f"⚠️ Could not read {fname}")
        continue
    
    detect_document_corners_debug(img, fname)

print(f"\n\n🎉 Analysis complete! Check {DEBUG_DIR} for detailed visualizations.")
