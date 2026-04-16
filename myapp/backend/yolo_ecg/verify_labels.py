import os
import cv2

BASE_PATH = r"C:\Users\prana\Documents\GitHub\Heartline-AI\myapp\backend\yolo_ecg\crop_dataset"

def visualize(folder_type):
    img_dir = os.path.join(BASE_PATH, "images", folder_type)
    label_dir = os.path.join(BASE_PATH, "labels", folder_type)

    for file in os.listdir(img_dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            img_path = os.path.join(img_dir, file)
            txt_path = os.path.join(label_dir, file.replace(".jpg", ".txt").replace(".png", ".txt"))

            img = cv2.imread(img_path)
            h, w, _ = img.shape

            if not os.path.exists(txt_path):
                print(f"❌ Missing label: {file}")
                continue

            with open(txt_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                cls, x, y, bw, bh = map(float, line.strip().split())

                # Convert YOLO → pixel
                x_center = int(x * w)
                y_center = int(y * h)
                box_w = int(bw * w)
                box_h = int(bh * h)

                x1 = int(x_center - box_w / 2)
                y1 = int(y_center - box_h / 2)
                x2 = int(x_center + box_w / 2)
                y2 = int(y_center + box_h / 2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow("Verification", img)
            key = cv2.waitKey(0)

            if key == 27:  # ESC to exit
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Checking TRAIN...")
    visualize("train")

    print("Checking VAL...")
    visualize("val")