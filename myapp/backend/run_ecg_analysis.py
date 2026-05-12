"""
run_ecg_analysis.py  –  ECG Analysis Pipeline Orchestrator

Pipeline:
    1. YOLO predict    → generate annotated ECG detection image
    2. crop_ecg        → crop ECG region
    3. upscale         → upscale the cropped image
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2


def write_result(result_path: str, data: dict):
    """Persist a JSON result so the Node route can read it."""
    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)


def fallback_content_crop(image_path: Path, output_path: Path, pad: int = 16) -> bool:
    """Crop to non-white ECG content when box-based crop is unavailable."""
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Select darker-than-background pixels (grid/waveform/text) as content.
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    coords = cv2.findNonZero(mask)
    if coords is None:
        return False

    x, y, w, h = cv2.boundingRect(coords)
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, img.shape[1])
    y2 = min(y + h + pad, img.shape[0])

    cropped = img[y1:y2, x1:x2]
    if cropped.size == 0:
        return False

    return cv2.imwrite(str(output_path), cropped)


def main():
    parser = argparse.ArgumentParser(description="ECG Crop + Upscale Pipeline")
    parser.add_argument("--input", help="Path to input ECG image (optional; defaults to backend/uploads/ecg.jpg)")
    parser.add_argument("--run-id", help="Unique run identifier (optional)")
    parser.add_argument("--result-path", help="Path to write result JSON (optional)")
    args = parser.parse_args()

    # ── Paths ────────────────────────────────────────────────────────────
    backend_dir = Path(__file__).resolve().parent  # myapp/backend
    uploads_dir = backend_dir / "uploads"
    preprocessing_dir = backend_dir / "preprocessing"
    model_path = backend_dir / "yolo_ecg" / "runs" / "detect" / "train" / "weights" / "best.pt"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or "local"

    input_path = Path(args.input).resolve() if args.input else (uploads_dir / "ecg.jpg").resolve()
    cropped_path = (uploads_dir / f"cropped_{run_id}.jpg").resolve()
    final_output = (uploads_dir / f"processed_{run_id}.jpg").resolve()

    # Keep YOLO outputs separated per-run to avoid cross-talk between requests.
    yolo_output_dir = uploads_dir / "yolo_output" / run_id
    yolo_predict_dir = yolo_output_dir / "predict"
    yolo_output_dir.mkdir(parents=True, exist_ok=True)

    result_path = (
        Path(args.result_path).resolve()
        if args.result_path
        else (uploads_dir / f"result_{run_id}.json").resolve()
    )

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
            "details": "",
        })
        return

    # =====================================================================
    # STEP 1 – YOLO Detect/Predict
    # =====================================================================
    print("\n" + "=" * 60)
    print("STEP 1: YOLO Detect/Predict")
    print("=" * 60)

    try:
        from ultralytics import YOLO

        # Equivalent to:
        # yolo detect predict model=runs/detect/train/weights/best.pt source=<input_path>
        model = YOLO(str(model_path))
        model.predict(
            source=str(input_path),
            save=True,
            project=str(yolo_output_dir),
            name="predict",
            exist_ok=True,
        )
        print(f"YOLO predict complete. Output -> {yolo_predict_dir}")
    except Exception as e:
        write_result(str(result_path), {
            "success": False,
            "error": "YOLO predict failed",
            "details": str(e),
        })
        return

    predicted_images = [
        f for f in yolo_predict_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ] if yolo_predict_dir.exists() else []
    yolo_crop_source = predicted_images[0] if predicted_images else input_path

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

        crop_ecg_from_image(str(yolo_crop_source), str(cropped_path))
        if not cropped_path.exists():
            if not fallback_content_crop(input_path, cropped_path):
                # Final fallback for difficult inputs.
                shutil.copy2(str(input_path), str(cropped_path))
                print("Content crop failed. Using original ECG image as cropped output.")
            else:
                print("No box detected. Applied content-based fallback crop.")
        print(f"Crop complete. Output -> {cropped_path}")

    except Exception as e:
        write_result(str(result_path), {
            "success": False,
            "error": "ECG cropping failed",
            "details": str(e),
        })
        return

    # =====================================================================
    # STEP 3 – Upscale
    # =====================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Upscale Cropped ECG")
    print("=" * 60)

    try:
        cropped_img = cv2.imread(str(cropped_path))
        if cropped_img is None:
            write_result(str(result_path), {
                "success": False,
                "error": "Failed to read cropped ECG image",
                "details": str(cropped_path),
            })
            return

        scale_factor = 2.0
        h, w = cropped_img.shape[:2]
        upscaled = cv2.resize(
            cropped_img,
            (int(w * scale_factor), int(h * scale_factor)),
            interpolation=cv2.INTER_LANCZOS4,
        )
        cv2.imwrite(str(final_output), upscaled)
        print(f"Upscale complete. Output → {final_output}")

    except Exception as e:
        write_result(str(result_path), {
            "success": False,
            "error": "Upscaling failed",
            "details": str(e),
        })
        return

    # Read final image dimensions
    final_img = cv2.imread(str(final_output))
    h, w = final_img.shape[:2] if final_img is not None else (0, 0)

    write_result(str(result_path), {
        "success": True,
        "run_id": run_id,
        "width": w,
        "height": h,
        "processed_image": str(final_output),
        "cropped_image": str(cropped_path),
        "message": "ECG crop + upscale completed successfully",
    })

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print(f"Result JSON → {result_path}")
    print(f"Processed image → {final_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
