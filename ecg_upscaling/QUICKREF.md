# ECG Upscaling Module - Quick Reference

## Files Created

### Main Scripts
1. **bicubic_upscale.py** - Bicubic interpolation upscaling (4x4 neighborhood)
2. **lanczos_upscale.py** - Lanczos resampling upscaling (8x8 neighborhood) ⭐ Recommended
3. **compare_algorithms.py** - Compare both algorithms side-by-side with timing

### Support Files
4. **requirements.txt** - Python dependencies (opencv-python, numpy)
5. **examples.py** - Example usage scenarios
6. **README.md** - Comprehensive documentation
7. **QUICKREF.md** - This file

## Quick Start

### Best Quality (Recommended for Medical Images)
```bash
python lanczos_upscale.py
```
Output: `ECG_dataset/upscaled_images/`

### Fast with Good Quality
```bash
python bicubic_upscale.py
```
Output: `ECG_dataset/upscaled_images/`

### Compare Both Algorithms
```bash
python compare_algorithms.py
```
Output: `ECG_dataset/upscaled_bicubic/` and `ECG_dataset/upscaled_lanczos/`

## Configuration

To change the upscaling factor, edit the `scale_factor` variable in any script:

```python
# In main() function (around line 135)
scale_factor = 2.0  # Change to 3.0 for 3x or 4.0 for 4x
```

## Algorithms Explained

| Algorithm | Neighborhood | Speed | Quality | Use Case |
|-----------|-------------|-------|---------|----------|
| **Bicubic** | 4x4 pixels | Faster | Good | General upscaling |
| **Lanczos** | 8x8 pixels | Slower | Best | Medical images, fine details |

## Input/Output

- **Input**: `ECG_dataset/images/*.jpeg`
- **Output**: `ECG_dataset/upscaled_images/*.jpeg` (default)
- **Scale Factor**: 2x (configurable)

## Performance

For 11 images (~4000x3000 pixels each):
- **Bicubic**: ~15-20 seconds
- **Lanczos**: ~16-22 seconds
- **Quality**: Lanczos produces sharper, more detailed results

## When to Use Which?

Use **Lanczos** when:
- ✓ You need the highest quality
- ✓ Working with medical/diagnostic images
- ✓ Detail preservation is critical
- ✓ Processing time is not a constraint

Use **Bicubic** when:
- ✓ You need fast processing
- ✓ Quality is good enough
- ✓ Processing large batches quickly

## Python API Usage

```python
# Lanczos (Best Quality)
from lanczos_upscale import ECGLanczosUpscaler
upscaler = ECGLanczosUpscaler("input/", "output/", scale_factor=2.0)
stats = upscaler.upscale_all_images()

# Bicubic (Fast)
from bicubic_upscale import ECGUpscaler
upscaler = ECGUpscaler("input/", "output/", scale_factor=2.0)
stats = upscaler.upscale_all_images()
```

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Tips

1. For ECG images, **Lanczos is recommended** due to better preservation of waveform details
2. Both algorithms support any scale factor (1.5x, 2x, 3x, 4x, etc.)
3. Original images are never modified; upscaled versions are saved separately
4. Output directories are created automatically if they don't exist
5. Use `compare_algorithms.py` to see visual and performance differences

## Troubleshooting

**Error: "No images found"**
- Check that images are in `ECG_dataset/images/` folder
- Verify file extensions (.jpg, .jpeg, .png, etc.)

**Error: "Could not read image"**
- Ensure image files are not corrupted
- Check file permissions

**Out of memory**
- Reduce scale_factor
- Process images one at a time
- Reduce image resolution before upscaling
