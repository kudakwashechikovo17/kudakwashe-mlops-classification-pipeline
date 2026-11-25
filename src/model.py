from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_CLASS_NAMES = ("fresh", "half_fresh", "spoiled")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(
    os.environ.get("EARTHWISE_DATA_DIR", ROOT / "data" / "meat_freshness")
)
MODEL_DIR = Path(os.environ.get("EARTHWISE_MODEL_DIR", ROOT / "models"))
MODEL_PATH = MODEL_DIR / "earthwise_meat_model.pkl"
METRICS_PATH = MODEL_DIR / "earthwise_metrics.json"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"


def build_model(alpha: float = 1e-4):
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            max_iter=500,
            tol=1e-3,
            random_state=42,
        ),
    )


def evaluate(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str],
) -> dict:
    labels = list(range(len(class_names)))
    y_pred = model.predict(X)
    report = classification_report(
        y,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "precision_macro": float(
            precision_score(y, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=labels).tolist(),
        "class_names": list(class_names),
        "classification_report": report,
    }


def load_class_names() -> list[str]:
    if not CLASS_NAMES_PATH.exists():
        return list(DEFAULT_CLASS_NAMES)
    names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    if not isinstance(names, list) or len(names) < 2 or not all(
        isinstance(name, str) and name for name in names
    ):
        raise ValueError(f"Invalid class names in {CLASS_NAMES_PATH}")
    return names


def save_model(
    model: object,
    metrics: dict | None = None,
    class_names: Sequence[str] | None = None,
) -> None:
    names = list(class_names) if class_names is not None else load_class_names()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(model, handle)
    CLASS_NAMES_PATH.write_text(json.dumps(names, indent=2), encoding="utf-8")
    if metrics is not None:
        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_model() -> object:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Earthwise model not found at {MODEL_PATH}. "
            "Label images, prepare the train/test split, then run: python scripts/train.py"
        )
    with MODEL_PATH.open("rb") as handle:
        return pickle.load(handle)
