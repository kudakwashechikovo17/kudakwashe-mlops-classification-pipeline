from __future__ import annotations

from typing import Any

from src.model import load_class_names, load_model
from src.preprocessing import features_from_bytes


def predict_image(image_bytes: bytes) -> dict[str, Any]:
    model = load_model()
    class_names = load_class_names()
    features = features_from_bytes(image_bytes).reshape(1, -1)
    label_id = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    top_indices = probabilities.argsort()[::-1][:3]
    return {
        "prediction": class_names[label_id],
        "confidence": float(probabilities[label_id]),
        "top_predictions": [
            {
                "label": class_names[int(index)],
                "confidence": float(probabilities[index]),
            }
            for index in top_indices
        ],
    }
