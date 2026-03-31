"""
Combined ECG Image Processing Pipeline
---------------------------------------
This script processes ECG images through a complete pipeline:
1. Upscaling using Lanczos Resampling
2. Brightness Augmentation using CLAHE
3. Contrast Augmentation using AGCWD

Input: Images from ECG_dataset/images folder
Output: Fully processed images in ECG_dataset/processed_images folder
"""

import cv2
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


class ECGImagePipeline:
    """
    Complete pipeline for ECG image processing
    """
    
    def __init__(self, input_dir, output_dir, scale_factor=2.0, 
                 clahe_clip_limit=2.0, clahe_tile_size=(8, 8),
                 agcwd_alpha=0.5, save_intermediate=False):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.scale_factor = scale_factor
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.agcwd_alpha = agcwd_alpha
        self.save_intermediate = save_intermediate
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create intermediate directories if needed
        if self.save_intermediate:
            self.upscaled_dir = self.output_dir / "intermediate_upscaled"
            self.brightness_dir = self.output_dir / "intermediate_brightness"
            self.upscaled_dir.mkdir(parents=True, exist_ok=True)
            self.brightness_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported image formats
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # CLAHE object
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, 
            tileGridSize=self.clahe_tile_size
        )
        
        # Statistics
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0
        }
    
    def step1_upscale(self, img):
        """        
        Args:
            img (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Upscaled image
        """
        height, width = img.shape[:2]
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)
        
        upscaled_img = cv2.resize(
            img, 
            (new_width, new_height), 
            interpolation=cv2.INTER_LANCZOS4
        )
        
        return upscaled_img
    
    def step2_brightness_clahe(self, img):
        """        
        Args:
            img (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Brightness-augmented image
        """
        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Apply CLAHE on brightness channel
        v_clahe = self.clahe.apply(v)
        
        # Global normalization
        v_norm = cv2.normalize(v_clahe, None, 0, 255, cv2.NORM_MINMAX)
        
        # Merge back and convert to BGR
        hsv_aug = cv2.merge([h, s, v_norm])
        augmented = cv2.cvtColor(hsv_aug, cv2.COLOR_HSV2BGR)
        
        return augmented
    
    def step3_contrast_agcwd(self, image):
        """        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Contrast-augmented image
        """
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Normalize to [0, 1]
        img_float = v.astype(np.float32) / 255.0
        
        # Calculate PDF and CDF
        hist, _ = np.histogram(v.flatten(), 256, [0, 256])
        pdf = hist / hist.sum()
        cdf = np.cumsum(pdf)
        
        # Calculate Weighting Distribution
        wd = cdf ** self.agcwd_alpha
        
        # Calculate Adaptive Gamma
        gamma = 1 - wd[v]
        gamma = np.clip(gamma, 0.01, 10.0)
        
        # Apply Gamma Correction
        enhanced_float = np.power(img_float, gamma)
        enhanced_uint8 = np.clip(enhanced_float * 255.0, 0, 255).astype(np.uint8)
        
        # Merge H, S with enhanced V and convert back to BGR
        enhanced_hsv = cv2.merge((h, s, enhanced_uint8))
        enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        
        return enhanced_image
    
    def process_image(self, image_path):
        """        
        Args:
            image_path (Path): Path to the input image
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Processing: {image_path.name}")
            
            # Read the image
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"  ❌ Error: Could not read image")
                return False
            
            original_size = img.shape[:2]
            print(f"  Original size: {original_size[1]}x{original_size[0]}")
            
            # Step 1: Upscale
            print(f"  → Step 1/3: Upscaling with Lanczos...")
            img_upscaled = self.step1_upscale(img)
            upscaled_size = img_upscaled.shape[:2]
            print(f"    New size: {upscaled_size[1]}x{upscaled_size[0]}")
            
            if self.save_intermediate:
                intermediate_path = self.upscaled_dir / image_path.name
                cv2.imwrite(str(intermediate_path), img_upscaled)
            
            # Step 2: Brightness augmentation (CLAHE)
            print(f"  → Step 2/3: Applying CLAHE brightness augmentation...")
            img_brightness = self.step2_brightness_clahe(img_upscaled)
            
            if self.save_intermediate:
                intermediate_path = self.brightness_dir / image_path.name
                cv2.imwrite(str(intermediate_path), img_brightness)
            
            # Step 3: Contrast augmentation (AGCWD)
            print(f"  → Step 3/3: Applying AGCWD contrast augmentation...")
            img_final = self.step3_contrast_agcwd(img_brightness)
            
            # Save final result
            output_path = self.output_dir / image_path.name
            cv2.imwrite(str(output_path), img_final)
            
            print(f"  ✅ Saved to: {output_path.name}\n")
            return True
            
        except Exception as e:
            print(f"  ❌ Error processing {image_path.name}: {str(e)}\n")
            return False
    
    def process_all_images(self):
        """        
        Returns:
            dict: Statistics about the processing
        """
        # Get all image files
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(self.input_dir.glob(f"*{ext}"))
            image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ No images found in {self.input_dir}")
            return {"total": 0, "success": 0, "failed": 0}
        
        print(f"Found {len(image_files)} image(s) to process\n")
        print("="*70)
        
        # Process each image
        for img_path in image_files:
            if self.process_image(img_path):
                self.stats['success'] += 1
            else:
                self.stats['failed'] += 1
            self.stats['total'] += 1
        
        # Print summary
        print("="*70)
        print(f"\nPipeline Complete!")
        print(f"Total images: {self.stats['total']}")
        print(f"Successfully processed: {self.stats['success']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Output directory: {self.output_dir}")
        
        if self.save_intermediate:
            print(f"\nIntermediate results saved to:")
            print(f"  - Upscaled: {self.upscaled_dir}")
            print(f"  - Brightness: {self.brightness_dir}")
        
        return self.stats
    
    def visualize_pipeline(self, image_path, num_samples=1):
        """
         Args:
            image_path (Path): Path to the input image
            num_samples (int): Number of samples to visualize
        """
        # Read the image
        img_original = cv2.imread(str(image_path))
        if img_original is None:
            print(f"Error: Could not read image {image_path}")
            return
        
        # Process through each stage
        img_upscaled = self.step1_upscale(img_original)
        img_brightness = self.step2_brightness_clahe(img_upscaled)
        img_final = self.step3_contrast_agcwd(img_brightness)
        
        # Convert BGR to RGB for display
        img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        img_upscaled_rgb = cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2RGB)
        img_brightness_rgb = cv2.cvtColor(img_brightness, cv2.COLOR_BGR2RGB)
        img_final_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'ECG Image Processing Pipeline - {image_path.name}', fontsize=16)
        
        axes[0, 0].imshow(img_original_rgb)
        axes[0, 0].set_title(f'Original\n{img_original.shape[1]}x{img_original.shape[0]}')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img_upscaled_rgb)
        axes[0, 1].set_title(f'After Upscaling (Lanczos)\n{img_upscaled.shape[1]}x{img_upscaled.shape[0]}')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(img_brightness_rgb)
        axes[1, 0].set_title('After Brightness (CLAHE)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(img_final_rgb)
        axes[1, 1].set_title('Final (CLAHE + AGCWD)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Main function to run the complete pipeline
    """
    # Define paths
    script_dir = Path(__file__).parent
    
    # Input: ECG_dataset/images folder
    input_dir = script_dir / "ECG_dataset" / "images"
    
    # Output: ECG_dataset/processed_images folder
    output_dir = script_dir / "ECG_dataset" / "processed_images"
    
    print("="*70)
    print("ECG Image Processing Pipeline")
    print("="*70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"\nPipeline stages:")
    print("  1. Upscaling (Lanczos Resampling) - 2x scale")
    print("  2. Brightness Augmentation (CLAHE)")
    print("  3. Contrast Augmentation (AGCWD)")
    print("="*70)
    print()
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"❌ Error: Input directory does not exist: {input_dir}")
        return
    
    # Configuration
    scale_factor = 2.0  # 2x upscaling
    clahe_clip_limit = 2.0
    clahe_tile_size = (8, 8)
    agcwd_alpha = 0.5
    save_intermediate = True  # Set to False to save only final results
    
    # Create pipeline and process images
    pipeline = ECGImagePipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        scale_factor=scale_factor,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_size=clahe_tile_size,
        agcwd_alpha=agcwd_alpha,
        save_intermediate=save_intermediate
    )
    
    # Process all images
    stats = pipeline.process_all_images()
    
    return stats


if __name__ == "__main__":
    main()
