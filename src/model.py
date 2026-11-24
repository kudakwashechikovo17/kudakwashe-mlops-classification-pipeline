from __future__ import annotations

import json
import pickle
from pathlib import Path

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

CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "cifar10_model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"


def build_model() -> SGDClassifier:
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=100,
            tol=1e-3,
            random_state=42,
        ),
    )


def evaluate(model: object, X: np.ndarray, y: np.ndarray) -> dict:
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balance_accuracy(y, y_pred)),
        "precision_macro": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": report,
    }


def balance_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return balanced_accuracy_score(y_true, y_pred)


def save_model(model: object, metrics: dict | None = None) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(model, handle)
    if metrics is not None:
        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_model() -> object:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python scripts/train.py"
        )
    with MODEL_PATH.open("rb") as handle:
        return pickle.load(handle)
