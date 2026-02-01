# ECG Image Augmentation Pipeline

This project implements a two-stage image augmentation pipeline for ECG (Electrocardiogram) images. It sequentially applies brightness correction followed by contrast enhancement to prepare the dataset for further analysis or machine learning tasks.

## 📂 Directory Structure

```text
Project2025_26/
├── run_pipeline.py                  # Master driver script
├── requirements.txt                 # Python dependencies
├── ecg_brightness-main/             # Stage 1: Brightness Augmentation
│   ├── clahe_augmentation.py        # Implementation of CLAHE
│   └── ...
├── ecg_contrast/                    # Stage 2: Contrast Augmentation
│   ├── agcwd_augment.py             # Implementation of AGCWD
│   └── ...
└── ECG_dataset/                     # Data Directory
    ├── images/                      # Input: Original Raw Images
    ├── augmented_clahe/             # Intermediate: Output of Stage 1
    └── augmented_agcwd/             # Final Output: Output of Stage 2
```

## 🔄 Workflow

The pipeline operates in a sequential manner:

1.  **Input**: Raw ECG images are placed in `ECG_dataset/images/`.
2.  **Stage 1 (Brightness)**: The system reads raw images and applies CLAHE (Contrast Limited Adaptive Histogram Equalization).
    *   *Input*: `ECG_dataset/images/`
    *   *Output*: `ECG_dataset/augmented_clahe/`
3.  **Stage 2 (Contrast)**: The system takes the brightness-corrected images from the previous step and applies AGCWD (Adaptive Gamma Correction with Weighting Distribution).
    *   *Input*: `ECG_dataset/augmented_clahe/`
    *   *Output*: `ECG_dataset/augmented_agcwd/`

## 🛠️ Code Functionality

### 1. Master Driver (`run_pipeline.py`)
This script automates the entire process. It executes the brightness script first, waits for it to complete, and then triggers the contrast script.

**Usage:**
```bash
python run_pipeline.py
```

### 2. Brightness Augmentation (`ecg_brightness-main/clahe_augmentation.py`)
This script uses **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to improve the local contrast of images.
*   **Method**: 
    1. Converts the image from BGR to HSV color space.
    2. Extracts the Value (V) channel (brightness).
    3. Applies CLAHE to the V channel with a clip limit of 2.0 and tile grid size of (8, 8).
    4. Normalizes the pixel values.
    5. Merges the channels back and converts to BGR.
*   **Purpose**: To correct uneven lighting and enhance local details.

### 3. Contrast Augmentation (`ecg_contrast/agcwd_augment.py`)
This script uses **AGCWD (Adaptive Gamma Correction with Weighting Distribution)** to enhance global contrast.
*   **Method**:
    1. Calculates the Probability Density Function (PDF) and Cumulative Distribution Function (CDF) of the image intensity.
    2. Computes a weighting distribution based on the CDF.
    3. Derives an adaptive gamma value for each pixel intensity.
    4. Applies the gamma correction.
*   **Purpose**: To improve the global dynamic range of the image, making the signal distinct from the background.

## 📦 Requirements
Ensure you have the necessary Python libraries installed:
```bash
pip install opencv-python numpy matplotlib
```
