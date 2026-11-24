from __future__ import annotations

import json
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
from fastapi.staticfiles import StaticFiles

from src.model import CIFAR10_LABELS, METRICS_PATH
from src.prediction import predict_image
from scripts.retrain import retrain

ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = ROOT / "data" / "uploads"
DATABASE_PATH = ROOT / "data" / "uploads.db"
STARTED_AT = time.time()

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="CIFAR-10 MLOps Image Classifier",
    description="Image prediction and incremental retraining API",
    version="1.0.0",
)

retrain_lock = threading.Lock()


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
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CIFAR-10 MLOps Console</title>
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
<header><h1>CIFAR-10 MLOps Console</h1><small id="uptime">Loading uptime...</small></header>
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
        <option value="0">0 - airplane</option><option value="1">1 - automobile</option>
        <option value="2">2 - bird</option><option value="3">3 - cat</option>
        <option value="4">4 - deer</option><option value="5">5 - dog</option>
        <option value="6">6 - frog</option><option value="7">7 - horse</option>
        <option value="8">8 - ship</option><option value="9">9 - truck</option>
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


@app.get("/health")
def health() -> dict:
    model_path = ROOT / "models" / "cifar10_model.pkl"
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - STARTED_AT, 2),
        "model_available": model_path.exists(),
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
    if label < 0 or label >= len(CIFAR10_LABELS):
        raise HTTPException(status_code=400, detail="Label must be between 0 and 9")

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
    return {"status": "saved", "saved_files": len(saved), "label": CIFAR10_LABELS[label], "files": saved}


@app.post("/retrain")
def trigger_retraining() -> dict:
    if retrain_lock.locked():
        raise HTTPException(status_code=409, detail="A retraining job is already running")

    def run_job() -> None:
        with retrain_lock:
            retrain(UPLOADS_DIR)

    thread = threading.Thread(target=run_job, daemon=True)
    thread.start()
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
    data_path = ROOT / "data"
    if (data_path / "test_images.npy").exists() and (data_path / "test_labels.npy").exists():
        images = np.load(data_path / "test_images.npy")
        labels = np.load(data_path / "test_labels.npy")
        class_counts = np.bincount(labels, minlength=len(CIFAR10_LABELS)).tolist()
        sample = images[:1000].astype(np.float32) / 255.0
        mean_brightness = float(sample.mean())
        mean_red = float(sample[:, :, :, 0].mean())
        mean_green = float(sample[:, :, :, 1].mean())
        mean_blue = float(sample[:, :, :, 2].mean())
    else:
        class_counts = [0] * len(CIFAR10_LABELS)
        mean_brightness = mean_red = mean_green = mean_blue = 0.0

    return {
        "class_names": CIFAR10_LABELS,
        "class_counts": class_counts,
        "feature_names": ["Mean brightness", "Red channel", "Green channel", "Blue channel"],
        "feature_values": [mean_brightness, mean_red, mean_green, mean_blue],
    }
