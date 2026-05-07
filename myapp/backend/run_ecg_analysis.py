"""
run_ecg_analysis.py  –  ECG Analysis Pipeline Orchestrator

Called by the Next.js API route (/api/upload) with:
    python run_ecg_analysis.py --input <path> --run-id <id> --result-path <path>

Pipeline:
    1. YOLO detection  → detect ECG lead bounding boxes
    2. crop_ecg        → crop the detected ECG region from the annotated image
    3. preprocessing   → upscale + CLAHE brightness + AGCWD contrast
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def write_result(result_path: str, data: dict):
    """Persist a JSON result so the Node route can read it."""
    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="ECG Analysis Pipeline")
    parser.add_argument("--input", required=True, help="Path to uploaded ECG image")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument("--result-path", required=True, help="Path to write result JSON")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    result_path = Path(args.result_path).resolve()
    run_id = args.run_id

    # ── Paths ────────────────────────────────────────────────────────────
    backend_dir = Path(__file__).resolve().parent          # myapp/backend
    yolo_ecg_dir = backend_dir / "yolo_ecg"
    preprocessing_dir = backend_dir / "preprocessing"
    model_path = yolo_ecg_dir / "runs" / "detect" / "train" / "weights" / "best.pt"

    # Per-run working directories (cleaned up at the end)
    run_work_dir = backend_dir / "uploads" / f"work_{run_id}"
    yolo_source_dir = run_work_dir / "yolo_input"
    yolo_output_dir = run_work_dir / "yolo_output"
    crop_output_dir = run_work_dir / "cropped"
    processed_output_dir = run_work_dir / "processed"

    # ── Sanity checks ────────────────────────────────────────────────────
    if not input_path.exists():
        write_result(str(result_path), {
            "success": False,
            "error": f"Input image not found: {input_path}",
            "details": "",
        })
        return

    if not model_path.exists():
        write_result(str(result_path), {
            "success": False,
            "error": f"YOLO model not found: {model_path}",
            "details": "Ensure best.pt exists at yolo_ecg/runs/detect/train/weights/best.pt",
        })
        return

    # ── Create working directories ───────────────────────────────────────
    for d in [yolo_source_dir, yolo_output_dir, crop_output_dir, processed_output_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy uploaded image into YOLO source directory
    yolo_input_image = yolo_source_dir / input_path.name
    shutil.copy2(str(input_path), str(yolo_input_image))

    # =====================================================================
    # STEP 1 – YOLO Detection
    # =====================================================================
    print("=" * 60)
    print("STEP 1: YOLO ECG Detection")
    print("=" * 60)

    try:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        results = model.predict(
            source=str(yolo_source_dir),
            save=True,
            project=str(yolo_output_dir),
            name="predict",
            exist_ok=True,
        )
        print(f"YOLO detection complete. Output → {yolo_output_dir / 'predict'}")
    except Exception as e:
        write_result(str(result_path), {
            "success": False,
            "error": "YOLO detection failed",
            "details": str(e),
        })
        return

    # =====================================================================
    # STEP 2 – Crop ECG Region
    # =====================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Crop ECG Region")
    print("=" * 60)

    try:
        # Add preprocessing dir to path so we can import crop_ecg
        sys.path.insert(0, str(preprocessing_dir))
        from crop_ecg import crop_ecg_from_image

        # The YOLO prediction saves annotated images in yolo_output_dir/predict/
        predict_dir = yolo_output_dir / "predict"
        predicted_images = [
            f for f in predict_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]

        if not predicted_images:
            write_result(str(result_path), {
                "success": False,
                "error": "No YOLO prediction images found",
                "details": f"Checked directory: {predict_dir}",
            })
            return

        cropped_path = crop_output_dir / predicted_images[0].name
        crop_ecg_from_image(str(predicted_images[0]), str(cropped_path))
        print(f"Crop complete. Output → {cropped_path}")

    except Exception as e:
        write_result(str(result_path), {
            "success": False,
            "error": "ECG cropping failed",
            "details": str(e),
        })
        return

    # =====================================================================
    # STEP 3 – Preprocessing Pipeline
    # =====================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Preprocessing Pipeline (Upscale → CLAHE → AGCWD)")
    print("=" * 60)

    try:
        from preprocessing_pipeline import ECGImagePipeline

        pipeline = ECGImagePipeline(
            input_dir=str(crop_output_dir),
            output_dir=str(processed_output_dir),
            scale_factor=2.0,
            clahe_clip_limit=2.0,
            clahe_tile_size=(8, 8),
            agcwd_alpha=0.5,
            save_intermediate=False,
        )

        pipeline.process_all_images()
        print(f"Preprocessing complete. Output → {processed_output_dir}")

    except Exception as e:
        write_result(str(result_path), {
            "success": False,
            "error": "Preprocessing pipeline failed",
            "details": str(e),
        })
        return

    # =====================================================================
    # Collect results
    # =====================================================================
    processed_images = [
        f for f in processed_output_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    if not processed_images:
        write_result(str(result_path), {
            "success": False,
            "error": "No processed images produced",
            "details": "",
        })
        return

    # Read final image dimensions
    import cv2
    final_img = cv2.imread(str(processed_images[0]))
    h, w = final_img.shape[:2] if final_img is not None else (0, 0)

    # Copy final processed image to a stable location in uploads/
    final_output = backend_dir / "uploads" / f"processed_{run_id}{processed_images[0].suffix}"
    shutil.copy2(str(processed_images[0]), str(final_output))

    write_result(str(result_path), {
        "success": True,
        "run_id": run_id,
        "width": w,
        "height": h,
        "processed_image": str(final_output),
        "message": "ECG analysis pipeline completed successfully",
    })

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print(f"Result JSON → {result_path}")
    print(f"Processed image → {final_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
