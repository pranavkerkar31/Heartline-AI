import cv2
import os

# -------- PATHS --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images", "train")
LABEL_DIR = os.path.join(BASE_DIR, "labels", "train")

os.makedirs(LABEL_DIR, exist_ok=True)

# -------- GLOBALS --------
points = []
orig_image = None
display_image = None
scale_x = 1.0
scale_y = 1.0
current_image_name = None


def resize_to_screen(image, max_width=1200, max_height=800):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    return cv2.resize(image, (int(w * scale), int(h * scale))), scale


def click_event(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        # Map click back to original image coords
        orig_x = int(x / scale_x)
        orig_y = int(y / scale_y)
        points.append((orig_x, orig_y))
        print(f"Point selected (orig): {orig_x}, {orig_y}")

        if len(points) == 2:
            save_bbox()


def save_bbox():
    global points, orig_image, current_image_name

    (x1, y1), (x2, y2) = points
    h, w = orig_image.shape[:2]

    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])

    # Draw on display image for confirmation
    disp = display_image.copy()
    cv2.rectangle(
        disp,
        (int(x_min * scale_x), int(y_min * scale_y)),
        (int(x_max * scale_x), int(y_max * scale_y)),
        (0, 255, 0),
        2,
    )
    cv2.imshow("Annotator", disp)

    # YOLO format
    x_center = ((x_min + x_max) / 2) / w
    y_center = ((y_min + y_max) / 2) / h
    box_width = (x_max - x_min) / w
    box_height = (y_max - y_min) / h

    label_path = os.path.join(
        LABEL_DIR, os.path.splitext(current_image_name)[0] + ".txt"
    )

    with open(label_path, "w") as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

    print(f"Saved label: {label_path}")
    print("Press 'n' for next image or 'q' to quit")

    points.clear()


def main():
    global orig_image, display_image, scale_x, scale_y, current_image_name

    images = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    for img_name in images:
        current_image_name = img_name
        img_path = os.path.join(IMAGE_DIR, img_name)
        orig_image = cv2.imread(img_path)

        if orig_image is None:
            continue

        display_image, scale = resize_to_screen(orig_image)
        scale_x = scale_y = scale

        print(f"\nAnnotating: {img_name}")
        print("Click TOP-LEFT then BOTTOM-RIGHT of ECG region")

        cv2.imshow("Annotator", display_image)
        cv2.setMouseCallback("Annotator", click_event)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("n"):
                break
            elif key == ord("q"):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
