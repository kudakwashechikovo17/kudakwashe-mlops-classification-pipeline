from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = (32, 32)
FEATURE_SIZE = (16, 16)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Apply EXIF orientation, RGB conversion, resizing, and normalization."""
    image = ImageOps.exif_transpose(image).convert("RGB").resize(IMAGE_SIZE)
    return np.asarray(image, dtype=np.float32) / 255.0


def image_to_features(image: Image.Image) -> np.ndarray:
    """Combine low-resolution RGB pixels and color histograms into one feature vector."""
    image_array = preprocess_image(image)
    small = np.asarray(
        Image.fromarray((image_array * 255).astype(np.uint8)).resize(FEATURE_SIZE),
        dtype=np.float32,
    )
    pixels = (small / 255.0).ravel()
    histograms = [
        np.histogram(image_array[:, :, channel], bins=8, range=(0.0, 1.0))[0]
        for channel in range(3)
    ]
    color_features = np.concatenate(histograms).astype(np.float32)
    color_features /= float(image_array.shape[0] * image_array.shape[1])
    return np.concatenate((pixels, color_features))


def image_from_bytes(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as image:
        return preprocess_image(image)


def features_from_bytes(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as image:
        return image_to_features(image)


def folder_images(
    folder: Path,
    class_names: Sequence[str] | None = None,
) -> Iterable[tuple[np.ndarray, int]]:
    """Read images from class-name folders or numeric retraining-label folders."""
    name_to_label = {name: index for index, name in enumerate(class_names or ())}
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parent_name = path.parent.name
        if parent_name.isdigit():
            label = int(parent_name)
        elif parent_name in name_to_label:
            label = name_to_label[parent_name]
        else:
            continue
        with Image.open(path) as image:
            yield image_to_features(image), label


def load_classification_split(
    folder: Path,
    class_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset split directory not found: {folder}")
    names = (
        list(class_names)
        if class_names is not None
        else sorted(path.name for path in folder.iterdir() if path.is_dir())
    )
    if len(names) < 2:
        raise ValueError(f"At least two labelled class folders are required in {folder}")

    samples = list(folder_images(folder, names))
    if not samples:
        raise ValueError(f"No supported, labelled images found under {folder}")
    features, labels = zip(*samples)
    return np.stack(features), np.asarray(labels, dtype=np.int64), names
