# Smart Camera with ROI Highlighting

A Python-based smart camera application that displays a highlighted Region of Interest (ROI) frame to guide users in positioning objects correctly before capturing images.

## Features

- **Live Camera Feed**: Real-time video display from your webcam
- **ROI Highlighting**: Green-framed area showing where to position your object
- **Dimmed Background**: Areas outside the ROI are semi-transparent to focus attention on the target area
- **Corner Markers**: Enhanced corner indicators for precise object alignment
- **Visual Feedback**: Flash effect when image is captured
- **On-screen Instructions**: Clear guidance displayed on the camera feed
- **Auto-save**: Captured images are automatically saved with timestamps
- **Image Counter**: Track how many images have been captured

## Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy

## Installation

1. Install the required dependencies:
```bash
pip install opencv-python numpy
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the script:
```bash
python smart_camera.py
```

### Controls

- **SPACE**: Capture image
- **Q**: Quit the application

### Instructions

1. Launch the application
2. Position your object within the green frame area
3. Ensure the object is centered and properly aligned with the corner markers
4. Press SPACE to capture the image
5. The image will be saved automatically in the `captured_images` folder
6. Continue capturing more images or press Q to quit

## Configuration

You can customize the camera behavior by modifying the `SmartCamera` initialization parameters:

```python
camera = SmartCamera(
    roi_percentage=0.7,  # ROI size (70% of frame)
    camera_index=0       # Camera device (0 = default camera)
)
```

### Parameters

- `roi_percentage`: Size of the ROI relative to the frame (0.1 to 1.0)
  - 0.5 = 50% of frame
  - 0.7 = 70% of frame (default)
  - 1.0 = full frame

- `camera_index`: Camera device index
  - 0 = Default/primary camera
  - 1, 2, etc. = Additional cameras if available

## Output

Captured images are saved in the `captured_images` folder with the following naming convention:
```
capture_YYYYMMDD_HHMMSS_XXXX.jpg
```

Where:
- `YYYYMMDD`: Date of capture
- `HHMMSS`: Time of capture
- `XXXX`: Sequential capture number (0001, 0002, etc.)

## Use Cases

- **Document Scanning**: Guide users to place documents within a specific frame
- **Product Photography**: Ensure consistent object positioning for e-commerce
- **ID Card Capture**: Align ID cards or documents properly
- **Quality Control**: Standardized image capture for inspection
- **Data Collection**: Capture images with consistent framing for datasets

## Advanced Features

### ROI Content Detection

The script includes a `check_roi_content()` method that can detect if there's sufficient content (edges) within the ROI. This can be enabled to validate object presence before allowing capture.

To enable validation, uncomment the relevant code in the `run()` method.

## Troubleshooting

### Camera Not Opening

- Check if another application is using the camera
- Try changing `camera_index` to 1 or 2
- Ensure you have camera permissions enabled

### Poor Performance

- Lower the camera resolution in the initialization
- Reduce the ROI overlay complexity

### Images Not Saving

- Check if the script has write permissions in the directory
- Ensure the `captured_images` folder exists or can be created

## License

This project is open source and available for educational and commercial use.

## Author

Created for Final Year Project 2025-26
