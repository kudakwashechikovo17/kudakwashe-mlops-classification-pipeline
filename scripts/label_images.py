from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageOps, ImageTk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import DATA_DIR, DEFAULT_CLASS_NAMES
from src.preprocessing import IMAGE_EXTENSIONS

OUTPUT_DIR = DATA_DIR / "labeled"
MANIFEST_PATH = OUTPUT_DIR / "labels.json"


def image_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ImageLabeler:
    def __init__(self, source: Path) -> None:
        if not source.is_dir():
            raise FileNotFoundError(f"Image source folder not found: {source}")
        self.source = source.resolve()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.manifest = (
            json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if MANIFEST_PATH.exists()
            else {}
        )
        self.paths = self._unique_images()
        self.index = 0
        self.root = tk.Tk()
        self.root.title("Earthwise Butchery - Image Labelling")
        self.root.geometry("1050x780")
        self.root.bind("1", lambda _event: self.label(DEFAULT_CLASS_NAMES[0]))
        self.root.bind("2", lambda _event: self.label(DEFAULT_CLASS_NAMES[1]))
        self.root.bind("3", lambda _event: self.label(DEFAULT_CLASS_NAMES[2]))
        self.root.bind("0", lambda _event: self.label("skip"))

        self.title = tk.Label(self.root, font=("Arial", 14))
        self.title.pack(pady=8)
        self.preview = tk.Label(self.root)
        self.preview.pack(expand=True)
        self.filename = tk.Label(self.root, wraplength=950)
        self.filename.pack(pady=5)

        buttons = tk.Frame(self.root)
        buttons.pack(pady=12)
        for number, name in enumerate(DEFAULT_CLASS_NAMES, start=1):
            tk.Button(
                buttons,
                text=f"{number}: {name.replace('_', ' ').title()}",
                width=18,
                command=lambda selected=name: self.label(selected),
            ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            buttons,
            text="0: Skip / unknown freshness",
            width=18,
            command=lambda: self.label("skip"),
        ).pack(side=tk.LEFT, padx=5)
        self.photo = None
        if self.paths:
            self.show_current()

    def _unique_images(self) -> list[Path]:
        unique: dict[str, Path] = {}
        for path in sorted(self.source.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            digest = image_digest(path)
            unique.setdefault(digest, path)
        return [
            path
            for path in unique.values()
            if image_digest(path) not in self.manifest
        ]

    def show_current(self) -> None:
        if self.index >= len(self.paths):
            messagebox.showinfo(
                "Labelling complete",
                f"All remaining images were reviewed.\nLabelled files are in {OUTPUT_DIR}",
            )
            self.root.destroy()
            return

        path = self.paths[self.index]
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((950, 620))
            self.photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.photo)
        self.filename.configure(text=str(path))
        self.title.configure(
            text=f"Image {self.index + 1} of {len(self.paths)} | "
            "Choose the known freshness label; skip images with no reliable label."
        )

    def label(self, selected: str) -> None:
        if self.index >= len(self.paths):
            return
        path = self.paths[self.index]
        digest = image_digest(path)
        if selected != "skip":
            destination_dir = OUTPUT_DIR / selected
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / f"{digest[:16]}{path.suffix.lower()}"
            shutil.copy2(path, destination)
        self.manifest[digest] = {
            "label": selected,
            "source_name": path.name,
        }
        MANIFEST_PATH.write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8"
        )
        self.index += 1
        self.show_current()

    def run(self) -> None:
        if not self.paths:
            messagebox.showinfo(
                "Nothing to label",
                f"No new supported images were found in {self.source}.",
            )
            self.root.destroy()
            return
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review source photos and label meat freshness classes."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=str(Path.home() / "Downloads" / "meat"),
        help="Folder containing source photos (default: ~/Downloads/meat)",
    )
    args = parser.parse_args()
    ImageLabeler(Path(args.folder)).run()


if __name__ == "__main__":
    main()
