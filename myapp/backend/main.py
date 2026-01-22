from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from PIL import Image, ExifTags
import numpy as np
import cv2
import io
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# COMMON — FIX EXIF ORIENTATION
# ==================================================
def fix_exif_orientation(pil_img: Image.Image) -> Image.Image:
    try:
        exif = pil_img._getexif()
        if exif is None:
            return pil_img

        orientation_key = None
        for key, value in ExifTags.TAGS.items():
            if value == "Orientation":
                orientation_key = key
                break

        if orientation_key is None:
            return pil_img

        orientation = exif.get(orientation_key)

        if orientation == 3:
            pil_img = pil_img.rotate(180, expand=True)
        elif orientation == 6:
            pil_img = pil_img.rotate(270, expand=True)
        elif orientation == 8:
            pil_img = pil_img.rotate(90, expand=True)

    except Exception:
        pass

    return pil_img


# ==================================================
# FAST DETECTION — FOR LIVE CAMERA PREVIEW
# ==================================================
def detect_paper_corners_fast(img_bgr):
    """
    Fast ECG paper detection.
    Used ONLY for live camera preview.
    """

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 80, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            return True, approx.reshape(4, 2).tolist()

    return False, []


# ==================================================
# LIVE DETECTION ENDPOINT (STEP 1 FOR CAMERA MODE)
# ==================================================
@app.post("/detect-paper")
async def detect_paper(file: UploadFile = File(...)):
    """
    Called repeatedly while camera is ON.
    Returns only:
    - found: bool
    - corners: 4 points (if found)
    """

    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil_image = fix_exif_orientation(pil_image)

    img_rgb = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    found, corners = detect_paper_corners_fast(img_bgr)

    return {
        "found": found,
        "corners": corners
    }


# ==================================================
# FINAL CAPTURE PIPELINE (AFTER AUTO / MANUAL CAPTURE)
# ==================================================
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Called ONCE after capture.
    High-quality processing.
    """

    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil_image = fix_exif_orientation(pil_image)

    img_rgb = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    os.makedirs("debug", exist_ok=True)

    # STEP 1
    cv2.imwrite("debug/step1_oriented.jpg", img_bgr)

    # STEP 2
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("debug/step2_grayscale.jpg", gray)

    # STEP 3
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite("debug/step3_blur.jpg", blur)

    # STEP 4
    edges = cv2.Canny(blur, 80, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    cv2.imwrite("debug/step4_edges.jpg", edges)

    # STEP 5
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    paper_corners = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            paper_corners = approx
            break

    debug = img_bgr.copy()

    if paper_corners is not None:
        cv2.drawContours(debug, [paper_corners], -1, (0, 255, 0), 4)
        for p in paper_corners:
            x, y = p[0]
            cv2.circle(debug, (x, y), 10, (0, 0, 255), -1)

        cv2.imwrite("debug/step5_corners.jpg", debug)

        return {
            "status": "ok",
            "corners": paper_corners.reshape(4, 2).tolist()
        }

    cv2.imwrite("debug/step5_corners.jpg", debug)
    return {
        "status": "failed",
        "corners": []
    }
