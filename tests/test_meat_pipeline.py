import json

import numpy as np
from PIL import Image

from scripts.train import train
from scripts.retrain import retrain
from scripts.import_freshness_dataset import import_dataset
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
    _write_example(tmp_path / "fresh", "fresh.png", (190, 160, 120))
    _write_example(tmp_path / "half_fresh", "half-fresh.png", (150, 100, 70))

    features, labels, names = load_classification_split(tmp_path)

    assert names == ["fresh", "half_fresh"]
    assert features.shape == (2, 792)
    assert labels.tolist() == [0, 1]


def test_training_saves_a_meat_model_and_class_metadata(tmp_path, monkeypatch):
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    colors = {
        "fresh": (190, 160, 120),
        "half_fresh": (150, 100, 70),
        "spoiled": (60, 70, 50),
    }
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
    assert metrics["class_names"] == ["fresh", "half_fresh", "spoiled"]
    assert json.loads((tmp_path / "classes.json").read_text()) == metrics["class_names"]
    assert (tmp_path / "model.pkl").is_file()
    assert (tmp_path / "metrics.json").is_file()


def test_retraining_updates_existing_model_and_test_metrics(tmp_path, monkeypatch):
    class_names = ["fresh", "half_fresh", "spoiled"]
    colors = [(190, 160, 120), (150, 100, 70), (60, 70, 50)]
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

    _write_example(tmp_path / "uploads" / "0", "new-fresh.png", (200, 170, 130))
    result = retrain(tmp_path / "uploads")

    assert result["status"] == "trained"
    assert result["samples_used"] == 1
    assert result["test_accuracy"] >= 0
    assert json.loads((tmp_path / "metrics.json").read_text())["class_names"] == class_names
    assert json.loads((tmp_path / "models" / "retraining_history.json").read_text())[0]["mode"] == "partial_fit"


def test_import_freshness_dataset_uses_csv_labels_and_keeps_validation_split(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "imported"
    class_columns = ("Fresh", "Half-Fresh", "Spoiled")
    class_names = ("fresh", "half_fresh", "spoiled")

    for split in ("train", "valid"):
        split_dir = source_dir / split
        split_dir.mkdir(parents=True)
        rows = []
        for label, (column, class_name) in enumerate(zip(class_columns, class_names)):
            filename = f"example-{label}.jpg"
            _write_example(split_dir, filename, (100 + label, 80, 60))
            rows.append(
                f"{filename},"
                + ",".join("1" if name == column else "0" for name in class_columns)
            )
        (split_dir / "_classes.csv").write_text(
            "filename, Fresh, Half-Fresh, Spoiled\n" + "\n".join(rows) + "\n",
            encoding="utf-8",
        )

    counts = import_dataset(source_dir, output_dir)

    assert counts == {
        "train": {"fresh": 1, "half_fresh": 1, "spoiled": 1},
        "test": {"fresh": 1, "half_fresh": 1, "spoiled": 1},
    }
    assert (output_dir / "test" / "half_fresh" / "example-1.jpg").is_file()
    assert (source_dir / "valid" / "example-1.jpg").is_file()
