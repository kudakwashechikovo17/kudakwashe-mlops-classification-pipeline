from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import DATA_DIR, DEFAULT_CLASS_NAMES
from src.preprocessing import IMAGE_EXTENSIONS

SOURCE_CLASSES = {
    "Fresh": "fresh",
    "Half-Fresh": "half_fresh",
    "Spoiled": "spoiled",
}
SOURCE_SPLITS = {"train": "train", "valid": "test"}


def _read_split(source_dir: Path, source_split: str) -> dict[str, list[Path]]:
    split_dir = source_dir / source_split
    labels_path = split_dir / "_classes.csv"
    if not split_dir.is_dir() or not labels_path.is_file():
        raise FileNotFoundError(
            f"Expected {source_split}/ and {source_split}/_classes.csv in {source_dir}"
        )

    image_paths = {
        path.name: path
        for path in split_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    result = {name: [] for name in DEFAULT_CLASS_NAMES}
    seen: set[str] = set()

    with labels_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = {column.strip() for column in reader.fieldnames or []}
        required_columns = {"filename", *SOURCE_CLASSES}
        if columns != required_columns:
            raise ValueError(
                f"Unexpected columns in {labels_path}: {sorted(columns)}"
            )

        for row in reader:
            values = {key.strip(): value.strip() for key, value in row.items() if key}
            filename = values.get("filename", "")
            if (
                not filename
                or Path(filename).name != filename
                or "/" in filename
                or "\\" in filename
            ):
                raise ValueError(f"Invalid image filename in {labels_path}: {filename!r}")
            if filename in seen:
                raise ValueError(f"Duplicate image label in {labels_path}: {filename}")
            seen.add(filename)
            if filename not in image_paths:
                raise ValueError(f"Labelled image is missing from {split_dir}: {filename}")

            active_classes = [
                SOURCE_CLASSES[source_name]
                for source_name in SOURCE_CLASSES
                if values.get(source_name) == "1"
            ]
            if len(active_classes) != 1 or any(
                values.get(source_name) not in {"0", "1"}
                for source_name in SOURCE_CLASSES
            ):
                raise ValueError(f"Invalid one-hot class labels for {filename}")
            result[active_classes[0]].append(image_paths[filename])

    if seen != set(image_paths):
        unlabeled = sorted(set(image_paths) - seen)
        raise ValueError(f"Images missing labels in {labels_path}: {unlabeled[:5]}")
    if any(not result[name] for name in DEFAULT_CLASS_NAMES):
        raise ValueError(f"Every class must occur in the {source_split} split")
    return result


def import_dataset(source_dir: Path, output_dir: Path = DATA_DIR) -> dict[str, dict[str, int]]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {source_dir}")
    if source_dir == output_dir or source_dir in output_dir.parents:
        raise ValueError("The output folder must not be inside the source dataset")
    for destination_split in SOURCE_SPLITS.values():
        if (output_dir / destination_split).exists():
            raise FileExistsError(
                f"{output_dir / destination_split} already exists; move it before importing."
            )

    source_data = {
        source_split: _read_split(source_dir, source_split)
        for source_split in SOURCE_SPLITS
    }
    counts: dict[str, dict[str, int]] = {"train": {}, "test": {}}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".meat-freshness-import-", dir=output_dir.parent
    ) as temporary:
        staging_dir = Path(temporary)
        for source_split, destination_split in SOURCE_SPLITS.items():
            for class_name, paths in source_data[source_split].items():
                class_dir = staging_dir / destination_split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                for path in paths:
                    shutil.copy2(path, class_dir / path.name)
                counts[destination_split][class_name] = len(paths)

        output_dir.mkdir(parents=True, exist_ok=True)
        for destination_split in SOURCE_SPLITS.values():
            os.replace(
                staging_dir / destination_split,
                output_dir / destination_split,
            )

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import the Roboflow meat freshness image dataset."
    )
    parser.add_argument("source", type=Path, help="Folder containing train/ and valid/")
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR,
        help=f"Output data directory (default: {DATA_DIR})",
    )
    args = parser.parse_args()
    counts = import_dataset(args.source, args.output)
    for split_name, class_counts in counts.items():
        print(f"{split_name}: {class_counts} ({sum(class_counts.values())} images)")


if __name__ == "__main__":
    main()
