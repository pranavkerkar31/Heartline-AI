"""
ECG Image Upscaling using Lanczos Resampling
---------------------------------------------
This script upscales ECG images using Lanczos resampling algorithm.
Input: Images from ECG_dataset/images folder
Output: Upscaled images saved to ECG_dataset/upscaled_images folder
"""

import cv2
import os
import numpy as np
from pathlib import Path


class ECGLanczosUpscaler:
    """
    Class for upscaling ECG images using Lanczos Resampling
    """
    
    def __init__(self, input_dir, output_dir, scale_factor=2.0):
        """
        Initialize the upscaler
        
        Args:
            input_dir (str): Path to input images directory
            output_dir (str): Path to output directory for upscaled images
            scale_factor (float): Factor by which to upscale (default: 2.0 = 2x upscaling)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.scale_factor = scale_factor
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported image formats
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # Statistics tracking
        self.total_original_pixels = 0
        self.total_upscaled_pixels = 0
    
    def upscale_image(self, image_path):
        """
        Upscale a single image using Lanczos resampling
        
        Args:
            image_path (Path): Path to the input image
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read the image
            img = cv2.imread(str(image_path))
            
            if img is None:
                print(f"Error: Could not read image {image_path.name}")
                return False
            
            # Get original dimensions
            height, width = img.shape[:2]
            
            # Calculate new dimensions
            new_width = int(width * self.scale_factor)
            new_height = int(height * self.scale_factor)
            
            # Calculate pixel counts
            original_pixels = width * height
            upscaled_pixels = new_width * new_height
            pixel_increase = upscaled_pixels - original_pixels
            
            # Track statistics
            self.total_original_pixels += original_pixels
            self.total_upscaled_pixels += upscaled_pixels
            
            print(f"Processing: {image_path.name}")
            print(f"  Original size: {width}x{height}")
            print(f"  Original pixel count: {original_pixels:,}")
            print(f"  New size: {new_width}x{new_height}")
            print(f"  Upscaled pixel count: {upscaled_pixels:,}")
            print(f"  Pixel increase: {pixel_increase:,} ({(pixel_increase/original_pixels)*100:.1f}%)")
            
            # Apply Lanczos resampling (INTER_LANCZOS4)
            # Lanczos uses an 8x8 neighborhood for high-quality resampling
            upscaled_img = cv2.resize(
                img, 
                (new_width, new_height), 
                interpolation=cv2.INTER_LANCZOS4
            )
            
            # Save the upscaled image
            output_path = self.output_dir / image_path.name
            cv2.imwrite(str(output_path), upscaled_img)
            
            print(f"  Saved to: {output_path.name}\n")
            return True
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {str(e)}")
            return False
    
    def upscale_all_images(self):
        """
        Upscale all images in the input directory
        
        Returns:
            dict: Statistics about the upscaling process
        """
        # Get all image files
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(self.input_dir.glob(f"*{ext}"))
            image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No images found in {self.input_dir}")
            return {"total": 0, "success": 0, "failed": 0}
        
        print(f"Found {len(image_files)} image(s) to process\n")
        print("="*60)
        
        # Process each image
        success_count = 0
        failed_count = 0
        
        for img_path in image_files:
            if self.upscale_image(img_path):
                success_count += 1
            else:
                failed_count += 1
        
        # Print summary
        print("="*60)
        print(f"\nUpscaling Complete!")
        print(f"Total images: {len(image_files)}")
        print(f"Successfully upscaled: {success_count}")
        print(f"Failed: {failed_count}")
        print(f"Output directory: {self.output_dir}")
        
        # Print pixel statistics
        if success_count > 0:
            print(f"\nPixel Statistics:")
            print(f"  Total original pixels: {self.total_original_pixels:,}")
            print(f"  Total upscaled pixels: {self.total_upscaled_pixels:,}")
            print(f"  Total pixel increase: {self.total_upscaled_pixels - self.total_original_pixels:,}")
            print(f"  Overall increase: {((self.total_upscaled_pixels - self.total_original_pixels) / self.total_original_pixels * 100):.1f}%")
        
        return {
            "total": len(image_files),
            "success": success_count,
            "failed": failed_count,
            "original_pixels": self.total_original_pixels,
            "upscaled_pixels": self.total_upscaled_pixels
        }


def count_image_pixels(image_path):
    """
    Count the number of pixels in an image
    
    Args:
        image_path (str or Path): Path to the image file
        
    Returns:
        dict: Dictionary with pixel count information including:
            - width: Image width in pixels
            - height: Image height in pixels
            - total_pixels: Total number of pixels (width × height)
            - megapixels: Total pixels in megapixels (MP)
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        height, width = img.shape[:2]
        total_pixels = width * height
        megapixels = total_pixels / 1_000_000
        
        return {
            "width": width,
            "height": height,
            "total_pixels": total_pixels,
            "megapixels": round(megapixels, 2)
        }
    except Exception as e:
        print(f"Error counting pixels: {str(e)}")
        return None


def main():
    """
    Main function to run the upscaling process
    """
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Input: ECG_dataset/images folder
    input_dir = project_root / "ECG_dataset" / "images"
    
    # Output: ECG_dataset/upscaled_images folder
    output_dir = project_root / "ECG_dataset" / "upscaled_images"
    
    # Set scale factor (can be modified as needed)
    # 2.0 = 2x upscaling (double the resolution)
    # 3.0 = 3x upscaling (triple the resolution)
    # 4.0 = 4x upscaling (quadruple the resolution)
    scale_factor = 2.0
    
    print("ECG Image Upscaling using Lanczos Resampling")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Scale factor: {scale_factor}x")
    print("="*60)
    print()
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Create upscaler and process images
    upscaler = ECGLanczosUpscaler(input_dir, output_dir, scale_factor)
    stats = upscaler.upscale_all_images()
    
    return stats


if __name__ == "__main__":
    main()
