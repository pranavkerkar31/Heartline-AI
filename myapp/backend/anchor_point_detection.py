import cv2
import numpy as np
import os

INPUT_DIR = "baseline_debug"     # ✅ baselines already drawn
OUTPUT_DIR = "anchor_debug"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------------------------
# EXTRACT BASELINES FROM RED LINES
# -------------------------------------------------
def extract_baselines_from_red(image):
    """
    Extract baseline y-coordinates from red horizontal lines
    """
    # red line → BGR (0,0,255)
    b, g, r = cv2.split(image)

    red_mask = (r > 200) & (g < 80) & (b < 80)

    ys, _ = np.where(red_mask)

    if len(ys) == 0:
        return []

    # cluster nearby y-values
    ys = np.sort(ys)
    baselines = [ys[0]]

    for y in ys[1:]:
        if abs(y - baselines[-1]) > 5:
            baselines.append(y)

    return baselines


# -------------------------------------------------
# VERTICAL ANCHOR POINTS
# -------------------------------------------------
def compute_vertical_anchors(baselines, image_height):
    anchors = []

    for i, y in enumerate(baselines):
        if i == 0:
            d = baselines[i + 1] - y
        elif i == len(baselines) - 1:
            d = y - baselines[i - 1]
        else:
            d = min(baselines[i + 1] - y, y - baselines[i - 1])

        upper = int(max(0, y - 0.7 * d))
        lower = int(min(image_height, y + 0.7 * d))

        anchors.append((upper, lower))

    return anchors


# -------------------------------------------------
# HORIZONTAL ANCHOR POINTS
# -------------------------------------------------
def compute_horizontal_anchors(binary, vertical_anchor):
    upper, lower = vertical_anchor
    roi = binary[upper:lower, :]

    projection = np.sum(roi > 0, axis=0)
    cols = np.where(projection > 0)[0]

    if len(cols) == 0:
        return 0, binary.shape[1]

    return cols[0], cols[-1]


# -------------------------------------------------
# MAIN PROCESS
# -------------------------------------------------
def process_folder():
    images = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for img_name in images:
        path = os.path.join(INPUT_DIR, img_name)
        img = cv2.imread(path)

        if img is None:
            continue

        h, w, _ = img.shape

        # --- extract baselines from red lines ---
        baselines = extract_baselines_from_red(img)

        if len(baselines) == 0:
            print(f"❌ No baselines found in {img_name}")
            continue

        # --- binary for signal extent ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        vertical_anchors = compute_vertical_anchors(baselines, h)

        debug = img.copy()

        for (upper, lower), y in zip(vertical_anchors, baselines):
            left, right = compute_horizontal_anchors(binary, (upper, lower))

            # draw anchor box
            cv2.rectangle(
                debug,
                (left, upper),
                (right, lower),
                (0, 255, 0),
                2
            )

            # redraw baseline (for clarity)
            cv2.line(debug, (0, y), (w, y), (0, 0, 255), 1)

        cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), debug)
        print(f"✅ Anchor points detected for {img_name}")

    print("\n🎉 Anchor point detection completed.")


if __name__ == "__main__":
    process_folder()
