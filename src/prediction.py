from __future__ import annotations

from pathlib import Path
from typing import Any

from src.model import CIFAR10_LABELS, load_model
from src.preprocessing import features_from_bytes


def predict_image(image_bytes: bytes) -> dict[str, Any]:
    model = load_model()
    features = features_from_bytes(image_bytes).reshape(1, -1)
    label_id = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    top_indices = probabilities.argsort()[::-1][:3]
    return {
        "prediction": CIFAR10_LABELS[label_id],
        "confidence": float(probabilities[label_id]),
        "top_predictions": [
            {
                "label": CIFAR10_LABELS[int(index)],
                "confidence": float(probabilities[index]),
            }
            for index in top_indices
        ],
    }
