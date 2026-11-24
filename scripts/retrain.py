from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import MODEL_PATH, METRICS_PATH, CIFAR10_LABELS, load_model, save_model
from src.preprocessing import folder_images


def retrain(upload_folder: Path) -> dict:
    model = load_model()
    samples = list(folder_images(upload_folder))
    if not samples:
        raise ValueError(
            "No valid images found. Put uploads in folders named with a label id (0-9). "
            "For example: new_data/3/cat.jpg"
        )

    X = np.asarray([features for features, _ in samples])
    y = np.asarray([label for _, label in samples])
    classifier = model.named_steps["sgdclassifier"] if hasattr(model, "named_steps") else model
    classes = np.arange(len(CIFAR10_LABELS))

    if hasattr(classifier, "partial_fit"):
        transformer = model[:-1] if hasattr(model, "__getitem__") and len(model) > 1 else None
        X_scaled = model[:-1].transform(X) if transformer is not None else X
        classifier.partial_fit(X_scaled, y, classes=classes)
        result = {"status": "trained", "samples_used": len(samples), "mode": "partial_fit"}
    else:
        raise RuntimeError("The active model does not support incremental retraining")

    save_model(model)
    summary = {
        **result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": str(MODEL_PATH),
    }
    history_path = ROOT / "models" / "retraining_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(summary)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="data/uploads")
    args = parser.parse_args()
    print(json.dumps(retrain(Path(args.folder)), indent=2))
