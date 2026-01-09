import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images", "train")
LABEL_DIR = os.path.join(BASE_DIR, "labels", "train")

images = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

for img_name in images:
    label_path = os.path.join(
        LABEL_DIR, os.path.splitext(img_name)[0] + ".txt"
    )

    if not os.path.exists(label_path):
        print(f"Missing label for {img_name}")
        continue

    img = cv2.imread(os.path.join(IMAGE_DIR, img_name))
    h, w = img.shape[:2]

    with open(label_path) as f:
        line = f.readline().strip().split()

    _, x_c, y_c, bw, bh = map(float, line)

    x1 = int((x_c - bw / 2) * w)
    y1 = int((y_c - bh / 2) * h)
    x2 = int((x_c + bw / 2) * w)
    y2 = int((y_c + bh / 2) * h)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Verify Labels", img)

    key = cv2.waitKey(0)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
