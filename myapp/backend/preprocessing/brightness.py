import cv2
import numpy as np
import os

INPUT_DIR = "cropped_ecg"
OUTPUT_DIR = "illumination_fixed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def adjust_gamma(image, gamma):

    invGamma = 1.0 / gamma

    table = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")

    return cv2.LUT(image, table)


def normalize_ecg_brightness(img):

    # -------- Step 1: Automatic Gamma Correction --------
    
    mean_brightness = np.mean(img)

    if mean_brightness < 120:
        gamma = 1.6      # brighten dark images
    elif mean_brightness > 180:
        gamma = 0.8      # slightly darken bright images
    else:
        gamma = 1.2      # mild adjustment

    gamma_corrected = adjust_gamma(img, gamma)

    # -------- Step 2: CLAHE for uneven lighting --------
    
    lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return result


# -------- Process all ECG images --------

for file in os.listdir(INPUT_DIR):

    path = os.path.join(INPUT_DIR, file)

    img = cv2.imread(path)

    if img is None:
        continue

    output = normalize_ecg_brightness(img)

    save_path = os.path.join(OUTPUT_DIR, file)

    cv2.imwrite(save_path, output)


print("ECG brightness normalization completed.")