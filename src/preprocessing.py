from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

IMAGE_SIZE = (32, 32)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert an uploaded image to a normalized RGB array."""
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def image_from_bytes(data: bytes) -> np.ndarray:
    return preprocess_image(Image.open(BytesIO(data)).copy())


def image_to_features(image: Image.Image) -> np.ndarray:
    """Create compact, deterministic features for a linear classifier."""
    array = preprocess_image(image)
    small = np.asarray(
        Image.fromarray((array * 255).astype(np.uint8)).resize((16, 16)),
        dtype=np.float32,
    )
    return (small / 255.0).ravel()


def features_from_bytes(data: bytes) -> np.ndarray:
    return image_to_features(Image.open(BytesIO(data)).copy())


def folder_images(folder: Path) -> Iterable[tuple[np.ndarray, int]]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in extensions:
            continue
        try:
            label = int(path.parent.name)
        except ValueError:
            continue
        yield image_to_features(Image.open(path).copy()), label
