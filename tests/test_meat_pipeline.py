import json

import numpy as np
from PIL import Image

from scripts.train import train
from scripts.retrain import retrain
from src.model import build_model, save_model
from src.preprocessing import image_to_features, load_classification_split
import src.model as model_module
import scripts.retrain as retrain_module


def _write_example(folder, filename, color):
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color=color).save(folder / filename)


def test_image_features_include_rgb_and_color_histograms():
    features = image_to_features(Image.new("RGB", (24, 24), color=(200, 80, 40)))
    assert features.shape == (792,)
    assert np.isfinite(features).all()


def test_classification_split_reads_named_image_folders(tmp_path):
    _write_example(tmp_path / "fish", "fish.png", (20, 40, 180))
    _write_example(tmp_path / "poultry", "chicken.png", (190, 160, 120))

    features, labels, names = load_classification_split(tmp_path)

    assert names == ["fish", "poultry"]
    assert features.shape == (2, 792)
    assert labels.tolist() == [0, 1]


def test_training_saves_a_meat_model_and_class_metadata(tmp_path, monkeypatch):
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    colors = {"fish": (20, 40, 180), "poultry": (190, 160, 120), "red_meat": (160, 30, 30)}
    for class_name, color in colors.items():
        for index in range(4):
            _write_example(train_dir / class_name, f"train-{index}.png", color)
        for index in range(2):
            _write_example(test_dir / class_name, f"test-{index}.png", color)

    monkeypatch.setattr(model_module, "MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(model_module, "CLASS_NAMES_PATH", tmp_path / "classes.json")
    monkeypatch.setattr(model_module, "METRICS_PATH", tmp_path / "metrics.json")

    metrics = train(train_dir, test_dir)

    assert metrics["test_samples"] == 6
    assert metrics["class_names"] == ["fish", "poultry", "red_meat"]
    assert json.loads((tmp_path / "classes.json").read_text()) == metrics["class_names"]
    assert (tmp_path / "model.pkl").is_file()
    assert (tmp_path / "metrics.json").is_file()


def test_retraining_updates_existing_model_and_test_metrics(tmp_path, monkeypatch):
    class_names = ["fish", "poultry", "red_meat"]
    colors = [(20, 40, 180), (190, 160, 120), (160, 30, 30)]
    train_features = []
    train_labels = []
    for label, color in enumerate(colors):
        for index in range(4):
            image = Image.new("RGB", (24, 24), color=color)
            train_features.append(image_to_features(image))
            train_labels.append(label)
            _write_example(
                tmp_path / "data" / "test" / class_names[label],
                f"test-{index}.png",
                color,
            )

    model = build_model()
    model.fit(np.stack(train_features), np.asarray(train_labels))
    monkeypatch.setattr(model_module, "MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(model_module, "CLASS_NAMES_PATH", tmp_path / "classes.json")
    monkeypatch.setattr(model_module, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(retrain_module, "ROOT", tmp_path)
    monkeypatch.setattr(retrain_module, "DATA_DIR", tmp_path / "data")
    save_model(model, class_names=class_names)

    _write_example(tmp_path / "uploads" / "0", "new-fish.png", (30, 55, 190))
    result = retrain(tmp_path / "uploads")

    assert result["status"] == "trained"
    assert result["samples_used"] == 1
    assert result["test_accuracy"] >= 0
    assert json.loads((tmp_path / "metrics.json").read_text())["class_names"] == class_names
    assert json.loads((tmp_path / "models" / "retraining_history.json").read_text())[0]["mode"] == "partial_fit"
