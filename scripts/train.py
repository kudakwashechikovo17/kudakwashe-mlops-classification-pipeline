from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import DATA_DIR, METRICS_PATH, build_model, evaluate, save_model
from src.preprocessing import (
    IMAGE_EXTENSIONS,
    load_classification_split,
    preprocess_image,
)

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"


def train(
    train_dir: Path = TRAIN_DIR,
    test_dir: Path = TEST_DIR,
) -> dict:
    X_train, y_train, class_names = load_classification_split(train_dir)
    X_test, y_test, test_classes = load_classification_split(
        test_dir, class_names=class_names
    )
    if class_names != test_classes:
        raise ValueError("Train and test class names do not match")

    counts = np.bincount(y_train, minlength=len(class_names))
    if np.any(counts < 3):
        raise ValueError(
            "Each training class needs at least three labelled images for "
            "3-fold hyperparameter tuning."
        )

    search = GridSearchCV(
        estimator=build_model(),
        param_grid={"sgdclassifier__alpha": [1e-5, 1e-4, 1e-3]},
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    metrics = evaluate(search.best_estimator_, X_test, y_test, class_names)
    metrics.update(
        {
            "best_params": search.best_params_,
            "training_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
        }
    )
    sample_images = []
    test_class_counts = []
    for class_name in class_names:
        class_dir = test_dir / class_name
        image_paths = [
            path
            for path in sorted(class_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        test_class_counts.append(len(image_paths))
        for path in image_paths[:100]:
            with Image.open(path) as image:
                sample_images.append(preprocess_image(image))
    if sample_images:
        image_array = np.stack(sample_images)
        metrics["image_insights"] = {
            "class_names": class_names,
            "class_counts": test_class_counts,
            "feature_names": [
                "Mean brightness",
                "Red channel",
                "Green channel",
                "Blue channel",
            ],
            "feature_values": [
                float(image_array.mean()),
                float(image_array[:, :, :, 0].mean()),
                float(image_array[:, :, :, 1].mean()),
                float(image_array[:, :, :, 2].mean()),
            ],
        }
    save_model(search.best_estimator_, metrics, class_names)

    print(f"Classes: {', '.join(class_names)}")
    print(f"Model saved; metrics saved to {METRICS_PATH}")
    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test macro F1: {metrics['f1_macro']:.4f}")
    return metrics


if __name__ == "__main__":
    train()
