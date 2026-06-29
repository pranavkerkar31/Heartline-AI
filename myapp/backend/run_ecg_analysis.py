"""
run_ecg_analysis.py  –  ECG Analysis Pipeline Orchestrator

Pipeline:
    1. YOLO predict    → generate annotated ECG detection image
    2. crop_ecg        → crop ECG region
    3. upscale         → upscale the cropped image
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import cv2


def write_result(result_path: str, data: dict):
    """Persist a JSON result so the Node route can read it."""
    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)


def report_progress(step: str, data: dict = None):
    """Print a structured progress message for the Node.js wrapper to capture."""
    msg = {"type": "progress", "step": step}
    if data:
        msg.update(data)
    print(f"PROGRESS:{json.dumps(msg)}", flush=True)


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


def auto_rotate_to_upright(image_path: Path) -> dict:
    """
    Detect text orientation by voting across all right-angle rotations.

    ECG reports contain small labels over graph paper, so raw Tesseract output can
    be empty unless the image is contrast-normalized first. Known report words are
    weighted more heavily than plain character count to avoid grid/trace noise.
    """
    try:
        import pytesseract
    except ImportError:
        print("pytesseract package is not installed. Run `pip install pytesseract` to enable auto-rotation.")
        return {"success": False, "error": "pytesseract not installed", "rotated": False}

    img = cv2.imread(str(image_path))
    if img is None:
        return {"success": False, "error": "Could not read image", "rotated": False}

    try:
        from PIL import Image, ImageOps

        with Image.open(image_path) as pil_img:
            exif_orientation = pil_img.getexif().get(274)
            if exif_orientation and exif_orientation != 1:
                transposed = ImageOps.exif_transpose(pil_img)
                save_kwargs = {}
                if image_path.suffix.lower() in (".jpg", ".jpeg"):
                    save_kwargs = {"quality": 95, "subsampling": 0}
                transposed.save(image_path, **save_kwargs)
                img = cv2.imread(str(image_path))
                print(f"Applied EXIF orientation tag: {exif_orientation}")
    except Exception as e:
        print(f"EXIF orientation normalization skipped: {e}")

    tesseract_cmd_candidates = [
        "tesseract",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\ProgramData\chocolatey\bin\tesseract.exe",
    ]

    tesseract_working_cmd = None
    for cmd in tesseract_cmd_candidates:
        try:
            pytesseract.pytesseract.tesseract_cmd = cmd
            pytesseract.get_tesseract_version()
            tesseract_working_cmd = cmd
            break
        except Exception:
            continue

    if not tesseract_working_cmd:
        print("\n" + "!" * 80)
        print("TESSERACT OCR IS NOT INSTALLED OR NOT FOUND.")
        print("Please install Tesseract OCR on Windows to enable automatic orientation detection:")
        print("1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Or run in an Administrator command prompt: winget install UB-Mannheim.TesseractOCR")
        print("!" * 80 + "\n")
        return {"success": False, "error": "Tesseract binary not found", "rotated": False}

    try:
        print(f"Running Robust Text-Based Orientation Detection on: {image_path.name}")

        rotations = [
            (0, None),
            (90, cv2.ROTATE_90_CLOCKWISE),
            (180, cv2.ROTATE_180),
            (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]
        keyword_pattern = re.compile(
            r"\b("
            r"ECG|EKG|REPORT|Patient|Technician|Confirmed|Room|Years|Male|Female|"
            r"mmHg|cm|kg|Medication|Diagnosis|Information|Department|Ref|Phys|"
            r"aVR|aVL|aVF|V[1-6]"
            r")\b",
            re.IGNORECASE,
        )
        config = "--oem 3 --psm 11 -c preserve_interword_spaces=1"

        def resize_for_ocr(gray_img):
            max_dim = 2400
            h_g, w_g = gray_img.shape
            if max(h_g, w_g) <= max_dim:
                return gray_img
            scale = max_dim / max(h_g, w_g)
            return cv2.resize(
                gray_img,
                (int(w_g * scale), int(h_g * scale)),
                interpolation=cv2.INTER_AREA,
            )

        def ocr_variants(candidate_img):
            gray_img = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)
            gray_img = resize_for_ocr(gray_img)
            gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
            denoised = cv2.GaussianBlur(gray_img, (3, 3), 0)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
            binary = cv2.adaptiveThreshold(
                clahe,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11,
            )
            return [clahe, binary]

        best = {
            "angle": 0,
            "score": -1,
            "chars": 0,
            "keywords": 0,
        }

        for angle, cv2_rot in rotations:
            candidate = img if cv2_rot is None else cv2.rotate(img, cv2_rot)
            angle_chars = 0
            angle_keywords = 0

            for variant in ocr_variants(candidate):
                try:
                    text = pytesseract.image_to_string(variant, config=config, timeout=10)
                except RuntimeError as e:
                    print(f"Tesseract skipped angle {angle} variant: {e}")
                    continue
                alnum_count = sum(c.isalnum() for c in text)
                keyword_count = len(keyword_pattern.findall(text))
                angle_chars = max(angle_chars, alnum_count)
                angle_keywords = max(angle_keywords, keyword_count)

            score = angle_chars + (angle_keywords * 250)
            print(
                f"Debug [Angle {angle} deg]: "
                f"chars={angle_chars}, keywords={angle_keywords}, score={score}"
            )

            if score > best["score"]:
                best = {
                    "angle": angle,
                    "score": score,
                    "chars": angle_chars,
                    "keywords": angle_keywords,
                }

        best_angle = best["angle"]
        max_alnum = best["chars"]
        print(
            "Robust Orientation Result: "
            f"Best Angle={best_angle} deg, Chars={best['chars']}, "
            f"Keywords={best['keywords']}, Score={best['score']}"
        )

        if best["score"] < 10:
            print("Low OCR confidence. Falling back to aspect-ratio.")
            h, w = img.shape[:2]
            best_angle = 90 if h > w else 0

        if best_angle != 0:
            if best_angle == 90:
                rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif best_angle == 180:
                rotated_img = cv2.rotate(img, cv2.ROTATE_180)
            elif best_angle == 270:
                rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            cv2.imwrite(str(image_path), rotated_img)
            print(f"Successfully rotated image {best_angle} degrees.")
            return {
                "success": True,
                "rotated": True,
                "angle": best_angle,
                "chars": max_alnum,
                "keywords": best["keywords"],
                "score": best["score"],
            }

        print("Image is already upright. No rotation needed.")
        return {
            "success": True,
            "rotated": False,
            "angle": 0,
            "chars": max_alnum,
            "keywords": best["keywords"],
            "score": best["score"],
        }

    except Exception as e:
        print(f"Error during Tesseract orientation detection: {e}")
        return {"success": False, "error": str(e), "rotated": False}


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

    report_progress("received", {"image": input_path.name})

    # ── Dynamic Orientation Correction (Tesseract & OpenCV) ──────────────
    print("Trusting Robust OCR Voting for complete orientation handling...")
    osd_result = auto_rotate_to_upright(input_path)
    if not osd_result.get("success"):
        print("Tesseract was not available or failed to detect orientation. Image left as is.")
        
    # Generate Debug Image
    debug_path = uploads_dir / f"orientation_{run_id}.jpg"
    dbg_img = cv2.imread(str(input_path))
    if dbg_img is not None:
        cv2.putText(dbg_img, f"Rotated: {osd_result.get('angle', 0)} deg", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10)
        cv2.putText(dbg_img, f"Confidence: {osd_result.get('chars', 0)} chars", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10)
        cv2.imwrite(str(debug_path), dbg_img)
        report_progress("orientation", {"image": debug_path.name})

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
        
        # Report YOLO output (the annotated image)
        yolo_annotated = [f for f in yolo_predict_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if yolo_annotated:
            # We need to make this accessible. For now just report the filename.
            # In STEP 1 YOLO saves to yolo_output/run_id/predict/input_id.jpg
            rel_yolo = f"yolo_output/{run_id}/predict/{yolo_annotated[0].name}"
            report_progress("yolo", {"image": rel_yolo})
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
        report_progress("crop", {"image": cropped_path.name})

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
        # Import the advanced preprocessing pipeline
        sys.path.insert(0, str(preprocessing_dir))
        from preprocessing_pipeline import ECGImagePipeline

        cropped_img = cv2.imread(str(cropped_path))
        if cropped_img is None:
            write_result(str(result_path), {
                "success": False,
                "error": "Failed to read cropped ECG image",
                "details": str(cropped_path),
            })
            return

        print("Applying advanced enhancements (Upscale + CLAHE + AGCWD)...")
        
        # Initialize pipeline (paths are dummy here as we use steps manually)
        pipeline = ECGImagePipeline(input_dir=uploads_dir, output_dir=uploads_dir)
        
        # Step 1: Advanced Upscale
        img_upscaled = pipeline.step1_upscale(cropped_img)
        
        # Step 2: Brightness Adjustment (CLAHE)
        img_brightness = pipeline.step2_brightness_clahe(img_upscaled)
        
        # Step 3: Contrast Enhancement (AGCWD)
        img_final = pipeline.step3_contrast_agcwd(img_brightness)
        
        cv2.imwrite(str(final_output), img_final)
        print(f"Enhancements complete. Output → {final_output}")
        report_progress("enhanced", {"image": final_output.name})

    except Exception as e:
        write_result(str(result_path), {
            "success": False,
            "error": "Image enhancement failed",
            "details": str(e),
        })
        return

    # Read final image dimensions
    final_img = cv2.imread(str(final_output))
    h, w = final_img.shape[:2] if final_img is not None else (0, 0)

    # =====================================================================
    # STEP 4 – Digitize ECG (Extract Signals to NPZ)
    # =====================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Digitize ECG (Extract Signals to NPZ)")
    print("=" * 60)

    npz_mv_path = None
    try:
        import numpy as np
        from scipy.signal import find_peaks
        
        # Ensure the backend directory is in path to import local ecg_digitizer
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
            
        from ecg_digitizer import run_digitization, DigitizerConfig

        # 1. Run main4.py logic to get baselines and px_per_mm
        gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)

        # --- Baseline Detection ---
        bg_gaussian = cv2.GaussianBlur(gray, (151, 151), 0)
        bg_median   = cv2.medianBlur(gray, 51)
        bg_combined = cv2.max(bg_gaussian, bg_median)
        diff = cv2.subtract(bg_combined, gray)
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        _, binary = cv2.threshold(diff_norm, 110, 255, cv2.THRESH_BINARY)
        
        kernel_open  = np.ones((2, 2), np.uint8)
        kernel_close = np.ones((3, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel_open,  iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        clean_final = np.zeros_like(binary)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 50:
                clean_final[labels == i] = 255

        row_sum = np.sum(clean_final, axis=1)
        peak_height = 0.20 * row_sum.max()
        peaks, _    = find_peaks(row_sum, height=peak_height, distance=h//6)
        if len(peaks) != 4:
            all_peaks, _ = find_peaks(row_sum, height=peak_height//2, distance=h//8)
            if len(all_peaks) >= 4:
                top4_idx = np.argsort(row_sum[all_peaks])[-4:]
                peaks    = np.sort(all_peaks[top4_idx])
        baselines = [float(p) for p in peaks]
        print(f"Detected baselines: {baselines}")

        # --- Grid Detection (px_per_mm) ---
        hist_vals  = np.bincount(gray.astype(np.uint8).ravel())
        paper_tone = int(np.argmax(hist_vals[180:]) + 180) if len(hist_vals) > 180 else 240
        GRID_HI    = paper_tone - 2
        GRID_LO    = paper_tone - 40
        grid_mask = ((gray >= GRID_LO) & (gray <= GRID_HI)).astype(np.float32)
        col_proj  = grid_mask.sum(axis=0)
        
        n = len(col_proj)
        p_arr = col_proj - col_proj.mean()
        F = np.fft.rfft(p_arr, n=2*n)
        acf = np.fft.irfft(F * np.conj(F))[:n].real
        acf = acf / (acf[0] + 1e-9)
        
        mm_per_px  = 250.0 / w
        px_per_mm_est  = 1.0 / mm_per_px
        p_min_auto = max(10, int(px_per_mm_est * 0.8))
        p_max_auto = min(150, int(px_per_mm_est * 8.0))
        
        search = acf[p_min_auto:p_max_auto+1]
        acf_peaks, props = find_peaks(search, height=0.03)
        if len(acf_peaks) > 0:
            order = np.argsort(props['peak_heights'])[::-1]
            best_period = acf_peaks[order[0]] + p_min_auto
            # The detected period is the major grid box (5mm), so divide by 5
            px_per_mm = float(round(best_period * 2) / 2) / 5.0
        else:
            px_per_mm = 20.0 # fallback
        print(f"Detected px_per_mm: {px_per_mm}")

        # 2. Save the segmentation mask and run digitizer on it
        out_dir = uploads_dir / "digitized" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        
        mask_path = uploads_dir / f"mask_{run_id}.png"
        
        # Convert the 1-channel mask to 3-channel BGR (black background, white signal)
        mask_bgr = cv2.cvtColor(clean_final, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(mask_path), mask_bgr)
        print(f"Saved segmentation mask -> {mask_path}")
        report_progress("mask", {"image": mask_path.name})
        
        config = DigitizerConfig(
            edge_label_zone_fraction=0.12,
            manual_px_per_mm=px_per_mm,
            manual_baselines=baselines
        )
        
        result = run_digitization(
            input_path=str(mask_path),
            output_dir=str(out_dir),
            config=config
        )
        
        npz_mv_path = str(result.npz_mv_path)
        print(f"Digitization complete. NPZ Output -> {npz_mv_path}")
        
        # Report the overlay image (it's in the out_dir)
        overlay_img = out_dir / "ecg_npz_overlay.png"
        if overlay_img.exists():
            rel_digitized = f"digitized/{run_id}/ecg_npz_overlay.png"
            report_progress("digitized", {"image": rel_digitized, "npz": f"digitized/{run_id}/ecg_signals_mv.npz"})

        # =====================================================================
        # STEP 5 – Predict ECG Diseases
        # =====================================================================
        print("\n" + "=" * 60)
        print("STEP 5: Predict ECG Diseases")
        print("=" * 60)
        
        prediction_data = None
        predict_script = backend_dir / "predict_digitized_ecg.py"
        if predict_script.exists():
            import subprocess
            try:
                cmd = [
                    sys.executable,
                    str(predict_script),
                    "--digitized-path",
                    str(out_dir)
                ]
                print(f"Running prediction: {' '.join(cmd)}")
                subprocess.run(cmd, cwd=str(backend_dir), capture_output=True, text=True, check=True)
                
                pred_json_path = out_dir / "prediction_result.json"
                if pred_json_path.exists():
                    with open(pred_json_path, "r", encoding="utf-8") as f:
                        prediction_data = json.load(f)
                    print(f"Prediction successful: {prediction_data.get('predicted_disease')} with confidence {prediction_data.get('confidence')}")
                    report_progress("predicted", {"prediction": prediction_data})
                else:
                    print("prediction_result.json was not created.")
                    report_progress("predicted", {"error": "Prediction result file not found"})
            except Exception as pe:
                print(f"Prediction script failed: {pe}")
                report_progress("predicted", {"error": f"Prediction script failed: {str(pe)}"})
        else:
            print("predict_digitized_ecg.py not found at", predict_script)
            report_progress("predicted", {"error": "predict_digitized_ecg.py not found"})

    except Exception as e:
        write_result(str(result_path), {
            "success": False,
            "error": "Digitization failed",
            "details": str(e),
        })
        return

    write_result(str(result_path), {
        "success": True,
        "run_id": run_id,
        "width": w,
        "height": h,
        "processed_image": str(final_output),
        "cropped_image": str(cropped_path),
        "npz_file": npz_mv_path,
        "prediction": prediction_data,
        "message": "ECG pipeline and digitization completed successfully",
    })

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print(f"Result JSON → {result_path}")
    print(f"Processed image → {final_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
