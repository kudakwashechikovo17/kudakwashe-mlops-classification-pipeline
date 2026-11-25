from __future__ import annotations

import json
import html
import logging
import sqlite3
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from src.model import (
    DATA_DIR,
    DEFAULT_CLASS_NAMES,
    METRICS_PATH,
    MODEL_PATH,
    load_class_names,
)
from src.prediction import predict_image
from scripts.retrain import retrain
from src.preprocessing import IMAGE_EXTENSIONS, preprocess_image

ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = DATA_DIR / "uploads"
DATABASE_PATH = DATA_DIR / "uploads.db"
STARTED_AT = time.time()

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Earthwise Butchery Meat Freshness Classifier",
    description="Image prediction and incremental retraining API",
    version="1.0.0",
)

retrain_lock = threading.Lock()
logger = logging.getLogger(__name__)


def init_database() -> None:
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                label INTEGER,
                stored_path TEXT NOT NULL,
                uploaded_at TEXT NOT NULL
            )
            """
        )


def record_upload(filename: str, label: int | None, stored_path: Path) -> None:
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            "INSERT INTO uploads (filename, label, stored_path, uploaded_at) VALUES (?, ?, ?, ?)",
            (
                filename,
                label,
                str(stored_path),
                datetime.now(timezone.utc).isoformat(),
            ),
        )


init_database()


def safe_image_name(name: str) -> str:
    return quote(Path(name).name, safe="")


@app.get("/", response_class=HTMLResponse)
def web_ui() -> str:
    try:
        class_names = load_class_names()
    except (OSError, ValueError):
        class_names = list(DEFAULT_CLASS_NAMES)
    label_options = "".join(
        f'<option value="{index}">{index} - {html.escape(name.replace("_", " ").title())}</option>'
        for index, name in enumerate(class_names)
    )
    page = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Earthwise Butchery MLOps Console</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background:#f4f7fb; color:#182230;}
    header { background:#172b4d; color:white; padding:18px 32px; }
    main { max-width:1100px; margin:auto; padding:24px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:18px;}
    .card { background:white; border-radius:10px; padding:18px; box-shadow:0 2px 10px #dfe6ef;}
    input, select, button { padding:9px; margin:5px 0; }
    button { background:#2563eb; color:white; border:none; border-radius:5px; cursor:pointer;}
    button:disabled { background:#94a3b8; }
    #preview { max-width:100%; max-height:260px; display:none; }
    pre { white-space:pre-wrap; }
  </style>
</head>
<body>
<header><h1>Earthwise Butchery MLOps Console</h1><small id="uptime">Loading uptime...</small></header>
<main>
  <div class="grid">
    <section class="card">
      <h2>Predict an image</h2>
      <input id="image" type="file" accept="image/*">
      <img id="preview" alt="Prediction preview">
      <button id="predict">Predict</button>
      <pre id="prediction">Choose an image.</pre>
    </section>
    <section class="card">
      <h2>Upload retraining data</h2>
      <label>Class label:</label>
      <select id="label">
        __LABEL_OPTIONS__
      </select>
      <input id="bulk" type="file" multiple accept="image/*,.zip">
      <button id="upload">Upload data</button>
      <button id="retrain">Trigger retraining</button>
      <pre id="retraining">No retraining run yet.</pre>
    </section>
  </div>
  <div class="grid" style="margin-top:18px">
    <section class="card"><h2>Class distribution</h2><canvas id="classChart"></canvas></section>
    <section class="card"><h2>Image feature insights</h2><canvas id="featureChart"></canvas></section>
  </div>
</main>
<script>
const uptimeEl=document.getElementById('uptime');
async function refreshHealth(){
 const r=await fetch('/health'); const d=await r.json();
 uptimeEl.textContent=`Status: ${d.status} | uptime: ${d.uptime_seconds}s | model: ${d.model_available?'ready':'missing'}`;
}
refreshHealth(); setInterval(refreshHealth,5000);
document.getElementById('image').onchange=e=>{
 const f=e.target.files[0]; if(!f)return;
 const p=document.getElementById('preview'); p.src=URL.createObjectURL(f); p.style.display='block';
};
document.getElementById('predict').onclick=async()=>{
 const f=document.getElementById('image').files[0]; if(!f)return alert('Choose an image');
 const fd=new FormData(); fd.append('file',f);
 document.getElementById('prediction').textContent='Predicting...';
 const r=await fetch('/predict',{method:'POST',body:fd}); document.getElementById('prediction').textContent=JSON.stringify(await r.json(),null,2);
};
document.getElementById('upload').onclick=async()=>{
 const files=document.getElementById('bulk').files; if(!files.length)return alert('Choose images or a ZIP');
 const fd=new FormData(); fd.append('label',document.getElementById('label').value);
 for(const f of files)fd.append('files',f);
 const r=await fetch('/upload',{method:'POST',body:fd}); document.getElementById('retraining').textContent=JSON.stringify(await r.json(),null,2);
};
document.getElementById('retrain').onclick=async()=>{
 document.getElementById('retraining').textContent='Retraining started...';
 const r=await fetch('/retrain',{method:'POST'}); document.getElementById('retraining').textContent=JSON.stringify(await r.json(),null,2);
};
fetch('/visualizations').then(r=>r.json()).then(d=>{
 new Chart(document.getElementById('classChart'),{type:'bar',data:{labels:d.class_names,datasets:[{label:'Test images',data:d.class_counts,backgroundColor:'#2563eb'}]}});
 new Chart(document.getElementById('featureChart'),{type:'bar',data:{labels:d.feature_names,datasets:[{label:'Average feature value',data:d.feature_values,backgroundColor:'#14b8a6'}]}});
}).catch(()=>{});
</script>
</body>
</html>"""
    return page.replace("__LABEL_OPTIONS__", label_options)


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - STARTED_AT, 2),
        "model_available": MODEL_PATH.exists(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        return predict_image(data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}") from exc


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...), label: int = Form(...)) -> dict:
    class_names = load_class_names()
    if label < 0 or label >= len(class_names):
        raise HTTPException(
            status_code=400,
            detail=f"Label must be between 0 and {len(class_names) - 1}",
        )

    saved = []
    label_folder = UPLOADS_DIR / str(label)
    label_folder.mkdir(parents=True, exist_ok=True)
    for upload in files:
        data = await upload.read()
        if not data:
            continue
        suffix = Path(upload.filename or "upload").suffix.lower()
        if suffix == ".zip":
            destination = UPLOADS_DIR / str(label) / f"archive_{int(time.time())}_{len(saved)}"
            destination.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(__import__("io").BytesIO(data)) as archive:
                for member in archive.namelist():
                    member_path = Path(member)
                    if member_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}:
                        continue
                    target = destination / member_path.name
                    target.write_bytes(archive.read(member))
                    record_upload(member_path.name, label, target)
                    saved.append(str(target.relative_to(ROOT)))
        elif suffix in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}:
            target = label_folder / f"{int(time.time()*1000)}_{safe_image_name(upload.filename or 'image')}"
            target.write_bytes(data)
            record_upload(upload.filename or "image", label, target)
            saved.append(str(target.relative_to(ROOT)))

    if not saved:
        raise HTTPException(status_code=400, detail="No valid images or ZIP file found")
    return {"status": "saved", "saved_files": len(saved), "label": class_names[label], "files": saved}


@app.post("/retrain")
def trigger_retraining() -> dict:
    if not retrain_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A retraining job is already running")

    def run_job() -> None:
        try:
            retrain(UPLOADS_DIR)
        except Exception as exc:
            logger.exception("Earthwise model retraining failed")
            history_path = ROOT / "models" / "retraining_history.json"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history = json.loads(history_path.read_text()) if history_path.exists() else []
            history.append(
                {
                    "status": "failed",
                    "error": str(exc),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        finally:
            retrain_lock.release()

    thread = threading.Thread(target=run_job, daemon=True)
    try:
        thread.start()
    except Exception:
        retrain_lock.release()
        raise
    return {"status": "retraining_started", "uploads_folder": str(UPLOADS_DIR)}


@app.get("/retrain/status")
def retraining_status() -> dict:
    history_path = ROOT / "models" / "retraining_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    return {
        "running": retrain_lock.locked(),
        "history": history[-10:],
    }


@app.get("/metrics")
def metrics() -> dict:
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=503, detail="Metrics are not available. Train the model first.")
    return json.loads(METRICS_PATH.read_text())


@app.get("/visualizations")
def visualizations() -> dict:
    class_names = load_class_names()
    test_dir = DATA_DIR / "test"
    class_counts = []
    feature_rows = []
    for class_name in class_names:
        class_dir = test_dir / class_name
        image_paths = (
            [
                path
                for path in sorted(class_dir.rglob("*"))
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if class_dir.is_dir()
            else []
        )
        class_counts.append(len(image_paths))
        for path in image_paths[:100]:
            with Image.open(path) as image:
                feature_rows.append(preprocess_image(image))

    if feature_rows:
        sample = np.stack(feature_rows)
        mean_brightness = float(sample.mean())
        mean_red = float(sample[:, :, :, 0].mean())
        mean_green = float(sample[:, :, :, 1].mean())
        mean_blue = float(sample[:, :, :, 2].mean())
    elif METRICS_PATH.exists():
        insights = json.loads(METRICS_PATH.read_text(encoding="utf-8")).get(
            "image_insights", {}
        )
        if insights.get("class_names") == class_names:
            class_counts = insights.get("class_counts", class_counts)
            feature_values = insights.get("feature_values", [0.0, 0.0, 0.0, 0.0])
            return {
                "class_names": class_names,
                "class_counts": class_counts,
                "feature_names": insights.get(
                    "feature_names",
                    ["Mean brightness", "Red channel", "Green channel", "Blue channel"],
                ),
                "feature_values": feature_values,
            }
        mean_brightness = mean_red = mean_green = mean_blue = 0.0
    else:
        mean_brightness = mean_red = mean_green = mean_blue = 0.0

    return {
        "class_names": class_names,
        "class_counts": class_counts,
        "feature_names": ["Mean brightness", "Red channel", "Green channel", "Blue channel"],
        "feature_values": [mean_brightness, mean_red, mean_green, mean_blue],
    }
