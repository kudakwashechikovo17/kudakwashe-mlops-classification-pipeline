from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import DATA_DIR, MODEL_PATH, evaluate, load_class_names, load_model, save_model
from src.preprocessing import folder_images, load_classification_split


def retrain(upload_folder: Path) -> dict:
    model = load_model()
    class_names = load_class_names()
    test_folder = DATA_DIR / "test"
    metrics = None
    if test_folder.is_dir():
        X_test, y_test, test_classes = load_classification_split(
            test_folder, class_names=class_names
        )
        if test_classes != class_names:
            raise ValueError("The retraining and test class names do not match")
    samples = list(folder_images(upload_folder, class_names))
    if not samples:
        raise ValueError(
            "No valid images found. Put uploaded images in numeric class folders "
            f"0-{len(class_names) - 1}, matching {class_names}."
        )

    X = np.asarray([features for features, _ in samples])
    y = np.asarray([label for _, label in samples])
    if np.any(y < 0) or np.any(y >= len(class_names)):
        raise ValueError(f"Upload labels must be between 0 and {len(class_names) - 1}")
    classifier = model.named_steps["sgdclassifier"] if hasattr(model, "named_steps") else model
    classes = np.arange(len(class_names))

    if hasattr(classifier, "partial_fit"):
        transformer = model[:-1] if hasattr(model, "__getitem__") and len(model) > 1 else None
        X_scaled = model[:-1].transform(X) if transformer is not None else X
        classifier.partial_fit(X_scaled, y, classes=classes)
        result = {"status": "trained", "samples_used": len(samples), "mode": "partial_fit"}
    else:
        raise RuntimeError("The active model does not support incremental retraining")

    if test_folder.is_dir():
        metrics = evaluate(model, X_test, y_test, class_names)
    save_model(model, metrics=metrics, class_names=class_names)
    summary = {
        **result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": str(MODEL_PATH),
        "evaluation_status": "updated" if metrics is not None else "unavailable; metrics remain from initial training",
        "metrics_current": metrics is not None,
    }
    if metrics is not None:
        summary["test_accuracy"] = metrics["accuracy"]
        summary["test_f1_macro"] = metrics["f1_macro"]
    history_path = ROOT / "models" / "retraining_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(summary)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="data/uploads")
    args = parser.parse_args()
    print(json.dumps(retrain(Path(args.folder)), indent=2))
