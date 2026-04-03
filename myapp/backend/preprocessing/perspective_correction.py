import cv2
import numpy as np
import os

INPUT_DIR = "../yolo_ecg/images/test"
OUTPUT_DIR = "straight_ecg"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def drawHoughLines(image, lines, output):
    out = image.copy()
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(output, out)


def intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    B = np.array([[rho1], [rho2]])

    x0, y0 = np.linalg.solve(A, B)
    return int(x0), int(y0)


def cyclic_intersection_pts(pts):
    center = np.mean(pts, axis=0)

    ordered = [
        pts[np.argmin(pts[:,0] + pts[:,1])],  # Top-left
        pts[np.argmin(pts[:,0] - pts[:,1])],  # Top-right
        pts[np.argmax(pts[:,0] + pts[:,1])],  # Bottom-right
        pts[np.argmax(pts[:,0] - pts[:,1])]   # Bottom-left
    ]
    return np.array(ordered, dtype="float32")


def straighten_ecg(image_path, out_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (0,0), fx=0.8, fy=0.8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    if lines is None:
        print("No lines detected:", image_path)
        return

    # Separate horizontal and vertical lines
    horizontal = []
    vertical = []

    for line in lines:
        rho, theta = line[0]
        if theta < np.pi/4 or theta > 3*np.pi/4:
            vertical.append(line)
        else:
            horizontal.append(line)

    # Take strongest 2 horizontal and 2 vertical lines
    horizontal = horizontal[:2]
    vertical = vertical[:2]

    # Find 4 intersection points
    pts = []
    for h in horizontal:
        for v in vertical:
            pts.append(intersection(h, v))

    pts = np.array(pts)
    pts = cyclic_intersection_pts(pts)

    # Perspective transform
    (tl, tr, br, bl) = pts

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth, 0],
        [maxWidth, maxHeight],
        [0, maxHeight]], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    cv2.imwrite(out_path, warped)


# Process folder
for file in os.listdir(INPUT_DIR):
    straighten_ecg(os.path.join(INPUT_DIR, file),
                   os.path.join(OUTPUT_DIR, file))

print("Done: ECG images straightened.")