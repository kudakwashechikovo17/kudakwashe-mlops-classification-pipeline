from __future__ import annotations

import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARCHIVE = DATA_DIR / "cifar-10-python.tar.gz"
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def download_and_extract() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not (DATA_DIR / "cifar-10-batches-py").exists():
        print("Downloading CIFAR-10...")
        urllib.request.urlretrieve(URL, ARCHIVE)
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            archive.extractall(DATA_DIR)

    batch_dir = DATA_DIR / "cifar-10-batches-py"
    train_images: list[np.ndarray] = []
    train_labels: list[int] = []
    for filename in [f"data_batch_{number}" for number in range(1, 6)]:
        with (batch_dir / filename).open("rb") as handle:
            batch = pickle.load(handle, encoding="bytes")
        train_images.append(batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        train_labels.extend(batch[b"labels"])

    with (batch_dir / "test_batch").open("rb") as handle:
        test_batch = pickle.load(handle, encoding="bytes")

    X_train = np.concatenate(train_images)
    y_train = np.asarray(train_labels)
    X_test = test_batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.asarray(test_batch[b"labels"])

    np.save(DATA_DIR / "train_images.npy", X_train)
    np.save(DATA_DIR / "train_labels.npy", y_train)
    np.save(DATA_DIR / "test_images.npy", X_test)
    np.save(DATA_DIR / "test_labels.npy", y_test)
    return X_train, y_train, X_test, y_test
