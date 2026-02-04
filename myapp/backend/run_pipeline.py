import subprocess
import sys
import os

def run_script(script_path):
    """Runs a python script as a subprocess."""
    print(f"🚀 Starting execution of: {script_path}")
    
    if not os.path.exists(script_path):
        print(f"❌ Error: Script not found at {script_path}")
        sys.exit(1)

    try:
        # Use the current Python interpreter to run the script
        result = subprocess.run([sys.executable, script_path], check=True)
        print(f"✅ Successfully finished: {script_path}\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error occurred while running {script_path}")
        print(f"Exit code: {e.returncode}")
        sys.exit(1)

def main():
    # Get the directory where this driver script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Run Brightness Augmentation (CLAHE)
    # Location: ecg_brightness-main/clahe_augmentation.py
    brightness_script = os.path.join(base_dir, "ecg_brightness", "clahe_augmentation.py")
    run_script(brightness_script)

    # 2. Run Contrast Augmentation (AGCWD)
    # Location: ecg_contrast/agcwd_augment.py
    # Note: This script reads the output from the previous step (augmented_clahe folder)
    contrast_script = os.path.join(base_dir, "ecg_contrast", "agcwd_augment.py")
    run_script(contrast_script)

    print("🎉 Full pipeline completed successfully!")

if __name__ == "__main__":
    main()
