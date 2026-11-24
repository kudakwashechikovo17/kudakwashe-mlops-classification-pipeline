from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_cifar10 import download_and_extract
from src.model import build_model, evaluate, save_model
from src.preprocessing import image_to_features


def train() -> None:
    X_train, y_train, X_test, y_test = download_and_extract()

    X_train_features = np.asarray(
        [image_to_features(Image.fromarray(image)) for image in X_train]
    )
    X_test_features = np.asarray(
        [image_to_features(Image.fromarray(image)) for image in X_test]
    )

    model = build_model()
    model.fit(X_train_features, y_train)
    metrics = evaluate(model, X_test_features, y_test)
    save_model(model, metrics)
    print("Model saved to models/cifar10_model.pkl")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    train()
