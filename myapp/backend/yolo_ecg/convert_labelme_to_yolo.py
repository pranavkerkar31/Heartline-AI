import os
import json
from pathlib import Path

# 🔥 Absolute path (RECOMMENDED for Windows)
BASE_PATH = "crop_dataset"


def convert_json_to_yolo(json_path, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_width = data['imageWidth']
    image_height = data['imageHeight']

    yolo_lines = []

    for shape in data['shapes']:
        label = shape['label']

        if label != "ecg":
            continue

        points = shape['points']

        # Labelme rectangle = 2 points
        x1, y1 = points[0]
        x2, y2 = points[1]

        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)

        # Convert to YOLO format
        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        yolo_lines.append(f"0 {x_center} {y_center} {width} {height}")

    txt_name = Path(json_path).stem + ".txt"
    txt_path = os.path.join(output_dir, txt_name)

    with open(txt_path, 'w') as f:
        f.write("\n".join(yolo_lines))


def process_folder(folder_type):
    json_dir = os.path.join(BASE_PATH, "labels", folder_type)
    output_dir = os.path.join(BASE_PATH, "labels", folder_type)

    for file in os.listdir(json_dir):
        if file.endswith(".json"):
            json_path = os.path.join(json_dir, file)
            convert_json_to_yolo(json_path, output_dir)


if __name__ == "__main__":
    print("🔄 Converting TRAIN labels...")
    process_folder("train")

    print("🔄 Converting VAL labels...")
    process_folder("val")

    print("✅ Conversion complete!")