# ECG Image Upscaling

This module provides image upscaling functionality for ECG images using multiple high-quality resampling algorithms.

## Algorithms Available

### 1. Bicubic Interpolation (`bicubic_upscale.py`)
- Uses 4x4 pixel neighborhood
- Good balance between quality and speed
- OpenCV's `INTER_CUBIC` method

### 2. Lanczos Resampling (`lanczos_upscale.py`)
- Uses 8x8 pixel neighborhood (windowed sinc function)
- Highest quality with superior edge preservation
- OpenCV's `INTER_LANCZOS4` method
- **Recommended for medical images requiring fine detail**

## Features

- **Multiple Algorithms**: Choose between Bicubic and Lanczos upscaling
- **Batch Processing**: Processes all images in the input directory
- **Flexible Scaling**: Configurable scale factor (default: 2x)
- **Multiple Format Support**: Supports JPG, JPEG, PNG, BMP, TIFF formats
- **Progress Tracking**: Displays processing status and statistics
- **Algorithm Comparison**: Compare performance of different methods
- **Pixel Counting**: Automatically counts and displays pixel statistics for original and upscaled images

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Bicubic Interpolation

Run the Bicubic upscaling script:

```bash
python bicubic_upscale.py
```

### Lanczos Resampling

Run the Lanczos upscaling script (recommended for best quality):

```bash
python lanczos_upscale.py
```

### Compare Algorithms

Compare both algorithms side-by-side:
use either upscaler class in your own scripts:

```python
# Bicubic Interpolation
from bicubic_upscale import ECGUpscaler

upscaler = ECGUpscaler(
    input_dir="path/to/input",
    output_dir="path/to/output",
    scale_factor=2.0
)
stats = upscaler.upscale_all_images()

# Lanczos Resampling
from lanczos_upscale import ECGLanczosUpscaler

upscaler = ECGLanczosUpscaler(
    input_dir="path/to/input",
    output_dir="path/to/output",
    scale_factor=2.0
)
stats = upscaler.upscale_all_images( own scripts:

```python
froAlgorithm Details

### Bicubic Interpolation
Bicubic interpolation considers the 16 nearest pixels (4x4 neighborhood) to calculate each new pixel value:
- Produces smooth results with good quality
- Faster processing speed
- Good for general upscaling tasks
- Uses OpenCV's `cv2.INTER_CUBIC` flag

### Lanczos Resampling
Lanczos resampling uses a windowed sinc function over 64 pixels (8x8 neighborhood):
- Highest quality among common resampling methods
- Superior edge preservation and sharpness
- Better handles fine details in ECG waveforms
- Minimal ringing artifacts
- **Recommended for medical imaging applications**
- Uses OpenCV's `cv2.INTER_LANCZOS4` flag

## Which Algorithm to Choose?

- **For best quality**: Use Lanczos (`lanczos_upscale.py`)
- **For speed with good quality**: Use Bicubic (`bicubic_upscale.py`)
- **For medical images**: Lanczos is recommended due to better detail preservation

# Process all images
stats = upscaler.upscale_all_images()

### Lanczos Upscaling
```
ECG Image Upscaling using Lanczos Resampling
============================================================
Input directory: ECG_dataset/images
Output directory: ECG_dataset/upscaled_images
Scale factor: 2.0x
============================================================

Found 11 image(s) to process

============================================================
Processing: test1.jpeg
  Original size: 4032x3024
  New size: 8064x6048
  Saved to: test1.jpeg

...

============================================================

Upscaling Complete!
Total images: 11
Successfully upscaled: 11
Failed: 0
Output directory: ECG_dataset/upscaled_images
```

### Algorithm Comparison
```
Speed Comparison:
  Bicubic is 5.2% faster than Lanczos

NOTES:
Bicubic: Good balance, 4x4 neighborhood
Lanczos: Highest quality, 8x8 neighborhood, better for medical images
Scale factor: 2.0x
============================================================

Found 11 image(s) to process

============================================================
Processing: test1.jpeg
  Original size: 800x600
  New size: 1600x1200
  Saved to: test1.jpeg

...

============================================================

Upscaling Complete!
Total images: 11
Successfully upscaled: 11
Failed: 0
Output directory: ECG_dataset/upscaled_images

Pixel Statistics:
  Total original pixels: 329,347,584
  Total upscaled pixels: 1,317,390,336
  Total pixel increase: 988,042,752
  Overall increase: 300.0%
```

## Pixel Counting

Both scripts automatically count and display pixel statistics during upscaling:

### Per-Image Statistics
Each image shows:
- Original dimensions and pixel count
- Upscaled dimensions and pixel count
- Pixel increase amount and percentage

### Overall Statistics
At the end of processing:
- Total original pixels across all images
- Total upscaled pixels across all images
- Total pixel increase
- Overall increase percentage

### Standalone Pixel Counter

Use the `count_image_pixels()` function to count pixels in any image:

```python
from bicubic_upscale import count_image_pixels

info = count_image_pixels("path/to/image.jpg")
print(f"Dimensions: {info['width']}x{info['height']}")
print(f"Total Pixels: {info['total_pixels']:,}")
print(f"Megapixels: {info['megapixels']} MP")
```

### Demo Script

Run the pixel counting demo:

```bash
python pixel_count_demo.py
```

This demonstrates:
- Counting pixels in individual images
- Comparing pixel counts at different scale factors
- Full upscaling with statistics

## Notes

- Original images are not modified; upscaled versions are saved separately
- The script automatically creates the output directory if it doesn't exist
- Supported image formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif
- Pixel counts help verify the quality improvement from upscaling
