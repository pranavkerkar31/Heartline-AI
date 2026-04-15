import cv2
import matplotlib.pyplot as plt
import os

INPUT_DIR = r"C:/Academics/FOURTH YEAR/FINAL YEAR PROJECT/Project2025_26/ECG_dataset/images"
OUTPUT_DIR = r"C:/Academics/FOURTH YEAR/FINAL YEAR PROJECT/Project2025_26/ECG_dataset/corrected_perspective"
COMPARE_DIR = r"C:/Academics/FOURTH YEAR/FINAL YEAR PROJECT/Project2025_26/ECG_dataset/comparison"

os.makedirs(COMPARE_DIR, exist_ok=True)

image_files = ['test1.jpeg', 'test2.jpeg', 'test3.jpeg']

for fname in image_files:
    original_path = os.path.join(INPUT_DIR, fname)
    corrected_path = os.path.join(OUTPUT_DIR, fname)
    
    original = cv2.imread(original_path)
    corrected = cv2.imread(corrected_path)
    
    if original is None or corrected is None:
        print(f"Could not load {fname}")
        continue
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'Comparison: {fname}', fontsize=16, fontweight='bold')
    
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Original\n{original.shape[1]}x{original.shape[0]}', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Corrected\n{corrected.shape[1]}x{corrected.shape[0]}', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(COMPARE_DIR, f'comparison_{fname}.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved comparison for {fname}")
    print(f"   Original size: {original.shape[1]}x{original.shape[0]}")
    print(f"   Corrected size: {corrected.shape[1]}x{corrected.shape[0]}")
    
    # Check if they're the same (no correction happened)
    if original.shape == corrected.shape:
        diff = cv2.absdiff(original, corrected)
        if diff.sum() < 1000:  # Very small difference
            print(f"   ⚠️ WARNING: Image appears UNCHANGED (no correction applied)")
    print()

print(f"\n📁 Comparisons saved to: {COMPARE_DIR}")
