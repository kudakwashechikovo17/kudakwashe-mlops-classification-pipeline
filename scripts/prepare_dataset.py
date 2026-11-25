from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import DATA_DIR, DEFAULT_CLASS_NAMES
from src.preprocessing import IMAGE_EXTENSIONS

LABELED_DIR = DATA_DIR / "labeled"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"


def prepare_dataset(test_fraction: float = 0.2, seed: int = 42) -> dict[str, dict[str, int]]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if TRAIN_DIR.exists() or TEST_DIR.exists():
        raise FileExistsError(
            "data/train or data/test already exists. Move or remove the generated "
            "split folders before preparing a new split."
        )

    splits: dict[str, dict[str, int]] = {"train": {}, "test": {}}
    for class_name in DEFAULT_CLASS_NAMES:
        source = LABELED_DIR / class_name
        if not source.is_dir():
            raise FileNotFoundError(
                f"Missing labelled class folder: {source}. "
                "Run scripts/label_images.py and review at least five photos per class."
            )
        images = sorted(
            path
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(images) < 5:
            raise ValueError(
                f"Class '{class_name}' has {len(images)} labelled images; "
                "at least five are required for a useful train/test split."
            )

        random.Random(seed).shuffle(images)
        test_count = max(1, round(len(images) * test_fraction))
        test_images = images[:test_count]
        train_images = images[test_count:]

        for split_name, split_images in (
            ("train", train_images),
            ("test", test_images),
        ):
            destination = (TRAIN_DIR if split_name == "train" else TEST_DIR) / class_name
            destination.mkdir(parents=True, exist_ok=True)
            for image in split_images:
                shutil.copy2(image, destination / image.name)
            splits[split_name][class_name] = len(split_images)

    return splits


if __name__ == "__main__":
    for split_name, counts in prepare_dataset().items():
        print(f"{split_name}: {counts}")
