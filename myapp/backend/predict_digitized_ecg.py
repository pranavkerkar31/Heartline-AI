"""
Predict ECG disease from digitized ECG output folders.

Expected digitized folder contents:
  ecg_signals_mv.npz
  Lead_I.png, Lead_II.png, ...

Usage from backend folder:
  python predict_digitized_ecg.py

Optional explicit usage:
  python predict_digitized_ecg.py ^
    --digitized-path outputs/a5340fea-c855-4cd9-8113-357796bb3064 ^
    --model-path models/ecg_cnn_lstm_digitized_model.keras ^
    --class-names models/ecg_class_names.json

The script writes prediction_result.json inside each digitized folder by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import signal
from tensorflow.keras.models import load_model


TARGET_LENGTH = 5000

# Use this if you trained with the new Colab notebook.
COLAB_LEAD_ORDER = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
    "RHYTHM",
]

# Use this only if your model was trained with the old notebook using sorted(ecg_leads).
OLD_SORTED_LEAD_ORDER = [
    "I", "II", "III",
    "RHYTHM",
    "V1", "V2", "V3", "V4", "V5", "V6",
    "aVF", "aVL", "aVR",
]


def get_lead_order(name: str) -> list[str]:
    if name == "colab":
        return COLAB_LEAD_ORDER
    if name == "old_sorted":
        return OLD_SORTED_LEAD_ORDER
    raise ValueError(f"Unsupported lead order: {name}")


def get_lead_data(npz_data: np.lib.npyio.NpzFile, lead_name: str):
    if lead_name in npz_data:
        return npz_data[lead_name]

    if lead_name == "RHYTHM" and "Rhythm" in npz_data:
        return npz_data["Rhythm"]

    return None


def remove_invalid_values(signal_data: np.ndarray) -> np.ndarray:
    signal_data = np.asarray(signal_data, dtype=np.float32).flatten()
    return np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)


def resample_signal(signal_data: np.ndarray, target_length: int = TARGET_LENGTH) -> np.ndarray:
    original_length = len(signal_data)

    if original_length == target_length:
        return signal_data.astype(np.float32)

    if original_length < 2:
        return np.zeros(target_length, dtype=np.float32)

    try:
        return signal.resample(signal_data, target_length).astype(np.float32)
    except Exception:
        x_old = np.linspace(0, 1, original_length)
        x_new = np.linspace(0, 1, target_length)
        return np.interp(x_new, x_old, signal_data).astype(np.float32)


def normalize_signal(signal_data: np.ndarray) -> np.ndarray:
    mean = float(np.mean(signal_data))
    std = float(np.std(signal_data))

    if std < 1e-8:
        return (signal_data - mean).astype(np.float32)

    return ((signal_data - mean) / std).astype(np.float32)


def preprocess_single_lead(lead_data: np.ndarray) -> np.ndarray:
    lead_data = remove_invalid_values(lead_data)
    lead_data = resample_signal(lead_data, TARGET_LENGTH)
    return normalize_signal(lead_data)


def load_and_process_npz(npz_path: Path, lead_order: list[str]) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    processed_leads = []
    missing_leads = []

    for lead_name in lead_order:
        lead_data = get_lead_data(data, lead_name)

        if lead_data is None:
            missing_leads.append(lead_name)
            continue

        processed_leads.append(preprocess_single_lead(lead_data))

    if missing_leads:
        raise ValueError(f"Missing leads in {npz_path}: {missing_leads}")

    ecg_array = np.stack(processed_leads, axis=0).astype(np.float32)
    expected_shape = (len(lead_order), TARGET_LENGTH)

    if ecg_array.shape != expected_shape:
        raise ValueError(f"Wrong ECG shape {ecg_array.shape}; expected {expected_shape}")

    # Model expects (batch, time, leads), so convert:
    # (13, 5000) -> (5000, 13) -> (1, 5000, 13)
    return np.expand_dims(ecg_array.transpose(1, 0), axis=0)


def find_digitized_npz_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.name != "ecg_signals_mv.npz":
            raise ValueError(f"Expected ecg_signals_mv.npz, got: {path}")
        return [path]

    direct_npz = path / "ecg_signals_mv.npz"
    if direct_npz.exists():
        return [direct_npz]

    return sorted(path.rglob("ecg_signals_mv.npz"))


def find_default_model_path() -> Path:
    preferred = Path("models") / "ecg_seresnet1d.keras"
    if preferred.exists():
        return preferred

    candidates = sorted(Path("models").glob("*.keras"))
    if candidates:
        return candidates[0]

    candidates = sorted(Path(".").glob("*.keras"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "Could not find a .keras model. Put it at "
        "models/ecg_cnn_lstm_digitized_model.keras or pass --model-path."
    )


def find_default_class_names_path() -> Path:
    preferred = Path("models") / "ecg_class_names.json"
    if preferred.exists():
        return preferred

    candidates = sorted(Path("models").glob("*class*.json"))
    if candidates:
        return candidates[0]

    candidates = sorted(Path(".").glob("*class*.json"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "Could not find class names JSON. Put it at "
        "models/ecg_class_names.json or pass --class-names."
    )


def load_class_names(class_names_path: Path) -> list[str]:
    with class_names_path.open("r", encoding="utf-8") as f:
        class_names = json.load(f)

    if not isinstance(class_names, list) or not class_names:
        raise ValueError("Class names JSON must contain a non-empty list.")

    return [str(label) for label in class_names]


def predict_one_folder(
    npz_path: Path,
    model,
    class_names: list[str],
    lead_order: list[str],
    write_json: bool,
) -> dict:
    model_input = load_and_process_npz(npz_path, lead_order)
    probabilities = model.predict(model_input, verbose=0)[0]

    if len(probabilities) != len(class_names):
        raise ValueError(
            f"Model returned {len(probabilities)} probabilities, "
            f"but class_names has {len(class_names)} labels."
        )

    predicted_index = int(np.argmax(probabilities))
    predicted_label = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    result = {
        "digitized_folder": str(npz_path.parent),
        "npz_path": str(npz_path),
        "predicted_disease": predicted_label,
        "confidence": confidence,
        "probabilities": {
            label: float(prob)
            for label, prob in zip(class_names, probabilities)
        },
    }

    if write_json:
        output_path = npz_path.parent / "prediction_result.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        result["result_path"] = str(output_path)

    return result


def print_result(result: dict) -> None:
    print("\n" + "=" * 70)
    print(f"Folder    : {result['digitized_folder']}")
    print(f"Prediction: {result['predicted_disease']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print("Probabilities:")

    for label, prob in result["probabilities"].items():
        print(f"  {label}: {prob:.6f}")

    if "result_path" in result:
        print(f"Saved JSON: {result['result_path']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict ECG disease from digitized ecg_signals_mv.npz folders."
    )
    parser.add_argument(
        "--digitized-path",
        default="outputs",
        help="Path to one digitized folder, one ecg_signals_mv.npz file, or parent backend/outputs folder. Default: outputs",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to trained .keras model. Default: auto-detect from models/*.keras",
    )
    parser.add_argument(
        "--class-names",
        default=None,
        help="Path to ecg_class_names.json saved during training. Default: auto-detect from models/ecg_class_names.json",
    )
    parser.add_argument(
        "--lead-order",
        choices=["colab", "old_sorted"],
        default="colab",
        help="Use 'colab' for the new notebook, or 'old_sorted' for old sorted(ecg_leads) models.",
    )
    parser.add_argument(
        "--no-write-json",
        action="store_true",
        help="Print predictions only; do not write prediction_result.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    digitized_path = Path(args.digitized_path)
    model_path = Path(args.model_path) if args.model_path else find_default_model_path()
    class_names_path = Path(args.class_names) if args.class_names else find_default_class_names_path()

    if not digitized_path.exists():
        raise FileNotFoundError(f"Digitized path not found: {digitized_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not class_names_path.exists():
        raise FileNotFoundError(f"Class names JSON not found: {class_names_path}")

    npz_files = find_digitized_npz_files(digitized_path)
    if not npz_files:
        raise FileNotFoundError(f"No ecg_signals_mv.npz found under: {digitized_path}")

    print(f"Loading model: {model_path}")
    model = load_model(model_path)
    class_names = load_class_names(class_names_path)
    lead_order = get_lead_order(args.lead_order)

    print(f"Class names: {class_names}")
    print(f"Lead order : {lead_order}")
    print(f"NPZ files  : {len(npz_files)}")

    results = []
    for npz_path in npz_files:
        result = predict_one_folder(
            npz_path=npz_path,
            model=model,
            class_names=class_names,
            lead_order=lead_order,
            write_json=not args.no_write_json,
        )
        results.append(result)
        print_result(result)

    print("\nDone.")


if __name__ == "__main__":
    main()
