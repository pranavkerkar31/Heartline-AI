import cv2
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class ECGImagePipeline:
    def __init__(self, input_dir, output_dir, scale_factor=2.0,
                 agcwd_alpha=0.5, save_intermediate=False):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.scale_factor = scale_factor
        self.agcwd_alpha = agcwd_alpha
        self.save_intermediate = save_intermediate
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create intermediate directories if needed
        if self.save_intermediate:
            self.upscaled_dir = self.output_dir / "intermediate_upscaled"
            self.upscaled_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported image formats
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        self.stats = {'total': 0, 'success': 0, 'failed': 0}
    
    def step1_upscale(self, img):
        height, width = img.shape[:2]
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)
        
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    def step3_contrast_agcwd(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        img_float = v.astype(np.float32) / 255.0
        hist, _ = np.histogram(v.flatten(), 256, [0, 256])
        pdf = hist / hist.sum()
        cdf = np.cumsum(pdf)
        
        wd = cdf ** self.agcwd_alpha
        gamma = 1 - wd[v]
        gamma = np.clip(gamma, 0.01, 10.0)
        
        enhanced_float = np.power(img_float, gamma)
        enhanced_uint8 = np.clip(enhanced_float * 255.0, 0, 255).astype(np.uint8)
        
        enhanced_hsv = cv2.merge((h, s, enhanced_uint8))
        return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    
    def process_image(self, image_path):
        try:
            print(f"Processing: {image_path.name}")
            
            img = cv2.imread(str(image_path))
            if img is None:
                print("Error reading image")
                return False
            
            img_upscaled = self.step1_upscale(img)
            img_final = self.step3_contrast_agcwd(img_upscaled)
            
            output_path = self.output_dir / image_path.name
            cv2.imwrite(str(output_path), img_final)
            
            print(f"Saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def process_all_images(self):
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(self.input_dir.glob(f"*{ext}"))
            image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No images found in {self.input_dir}")
            return
        
        print(f"Found {len(image_files)} images\n")
        
        for img_path in image_files:
            if self.process_image(img_path):
                self.stats['success'] += 1
            else:
                self.stats['failed'] += 1
            self.stats['total'] += 1
        
        print("\nProcessing Complete")
        print(self.stats)


def main():
    print("="*60)
    print("ECG Image Processing Pipeline")
    print("="*60)
    
    # FIXED PATHS
    input_dir = Path("cropped_ecg")
    output_dir = Path("processed_images")
    
    if not input_dir.exists():
        print(f"Input folder not found: {input_dir}")
        return
    
    pipeline = ECGImagePipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        scale_factor=2.0,
        agcwd_alpha=0.5,
        save_intermediate=False
    )
    
    pipeline.process_all_images()


if __name__ == "__main__":
    main()