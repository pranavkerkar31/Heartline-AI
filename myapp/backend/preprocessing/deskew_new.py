import cv2
import numpy as np
import os

print("🔥 Deskew pipeline running...")

# -------------------- PATH CONFIG --------------------

INPUT_DIR = "../yolo_ecg/crop_dataset/images/val"  # 👈 correct relative path
OUTPUT_DIR = "deskewed_ecg"                            # 👈 output folder

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- FUNCTIONS --------------------

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    maxWidth  = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))
    maxHeight = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-bl)))

    dst = np.array([
        [0, 0],
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def detect_document_corners(img):
    h, w = img.shape[:2]
    scale = 800 / max(h, w)

    small = cv2.resize(img, (int(w*scale), int(h*scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:5]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            return approx.reshape(4, 2) / scale

    # fallback
    margin = int(min(h, w) * 0.05)
    return np.array([
        [margin, margin],
        [w-margin, margin],
        [w-margin, h-margin],
        [margin, h-margin]
    ], dtype=np.float32)

def fix_orientation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rotations = [
        (0, None),
        (90, cv2.ROTATE_90_CLOCKWISE),
        (180, cv2.ROTATE_180),
        (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]

    best_score = -1e9
    best_img = img

    for _, rot_code in rotations:
        g = cv2.rotate(gray, rot_code) if rot_code else gray

        rh, rw = g.shape
        top_mean = np.mean(g[:int(rh*0.10), :])
        mid_mean = np.mean(g[int(rh*0.20):int(rh*0.70), :])

        score = top_mean - mid_mean

        if score > best_score:
            best_score = score
            best_img = cv2.rotate(img, rot_code) if rot_code else img

    return best_img

def process_image(input_path, output_path):
    print(f"\n📷 Processing: {input_path}")

    img = cv2.imread(input_path)

    if img is None:
        print(f"❌ Cannot read image: {input_path}")
        return

    corners = detect_document_corners(img)
    warped = four_point_transform(img, corners)

    h, w = warped.shape[:2]
    if h > w:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    final = fix_orientation(warped)

    cv2.imwrite(output_path, final)
    print(f"✅ Saved: {output_path}")

# -------------------- RUN --------------------

print("📂 INPUT DIR:", os.path.abspath(INPUT_DIR))

if not os.path.exists(INPUT_DIR):
    print("❌ INPUT_DIR does not exist!")
    exit()

files = os.listdir(INPUT_DIR)

if len(files) == 0:
    print("❌ No images found in INPUT_DIR!")
    exit()

for file in files:
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        inp = os.path.join(INPUT_DIR, file)

        name, ext = os.path.splitext(file)
        out = os.path.join(OUTPUT_DIR, f"{name}_deskewed{ext}")

        process_image(inp, out)

print("\n🎉 Done processing all images!")