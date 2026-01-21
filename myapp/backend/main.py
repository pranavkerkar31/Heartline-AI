from fastapi import FastAPI, UploadFile, File
from PIL import Image, ExifTags
import numpy as np
import cv2
import io
import os

app = FastAPI()

def fix_exif_orientation(pil_img):
    try:
        exif = pil_img._getexif()
        if exif is None:
            return pil_img

        for tag, value in ExifTags.TAGS.items():
            if value == "Orientation":
                orientation_tag = tag
                break

        orientation = exif.get(orientation_tag, None)

        if orientation == 3:
            pil_img = pil_img.rotate(180, expand=True)
        elif orientation == 6:
            pil_img = pil_img.rotate(270, expand=True)
        elif orientation == 8:
            pil_img = pil_img.rotate(90, expand=True)

    except Exception:
        pass

    return pil_img


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # 1️⃣ Read raw bytes
    image_bytes = await file.read()

    # 2️⃣ Load with PIL
    pil_image = Image.open(io.BytesIO(image_bytes))

    # 3️⃣ Fix EXIF orientation
    pil_image = fix_exif_orientation(pil_image)

    # 4️⃣ Convert to OpenCV format (BGR)
    img_np = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 🔍 DEBUG: save Step 1 output
    os.makedirs("debug", exist_ok=True)
    cv2.imwrite("debug/step1_oriented.jpg", img_bgr)

    return {
        "status": "ok",
        "width": img_bgr.shape[1],
        "height": img_bgr.shape[0]
    }
