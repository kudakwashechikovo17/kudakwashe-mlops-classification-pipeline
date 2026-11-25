from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import MODEL_DIR, MODEL_PATH


def main() -> None:
    source_models = ROOT / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        if not (source_models / MODEL_PATH.name).is_file():
            raise FileNotFoundError(
                f"Earthwise model not found at {MODEL_PATH} or {source_models}. "
                "Train the model before starting the deployment."
            )
        for filename in (
            "earthwise_meat_model.pkl",
            "class_names.json",
            "earthwise_metrics.json",
        ):
            source = source_models / filename
            if source.is_file():
                shutil.copy2(source, MODEL_DIR / filename)

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
