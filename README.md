# CIFAR-10 MLOps Classification Pipeline

An end-to-end, Dockerized image classification system that demonstrates data acquisition, preprocessing, model training, evaluation, prediction, incremental retraining, API deployment, monitoring, and load testing.

**Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), 50,000 training images and 10,000 test images across 10 image classes.

**Repository:** https://github.com/kudakwashechikovo17/kudakwashe-mlops-classification-pipeline

> Replace this section with your public deployment URL and YouTube demo link before submission.

- Production URL: `https://YOUR-CLOUD-APP.example`
- Video demo: `https://youtu.be/YOUR_VIDEO_ID`

## Project structure

```text
.
├── app/main.py               # FastAPI API and browser UI
├── data/                     # CIFAR-10, NumPy arrays, uploaded images
├── models/                   # .pkl model and metrics
├── notebook/project_name.ipynb
├── scripts/download_cifar10.py
├── scripts/train.py
├── scripts/retrain.py
├── src/model.py
├── src/prediction.py
├── src/preprocessing.py
├── locustfile.py
├── Dockerfile
├── docker-compose.yml
└── tests/test_api.py
```

## Features

- Image upload and three-label prediction response
- 10 CIFAR-10 image classes
- SQLite upload registry
- Bulk image and ZIP upload
- Triggered incremental retraining using the existing fitted model
- Retraining history and status endpoint
- Model uptime and health check
- Dataset class distribution and feature visualizations
- Docker deployment
- Locust flood testing
- Notebook evaluation with at least seven metrics and visualizations

## Local setup on Windows

### 1. Clone and create an environment

```powershell
git clone https://github.com/kudakwashechikovo17/kudakwashe-mlops-classification-pipeline.git
cd kudakwashe-mlops-classification-pipeline
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Python 3.12 is not available, Python 3.11 can also run this project.

### 2. Download and train

```powershell
python scripts/train.py
```

The script downloads CIFAR-10 if it is not already present, preprocesses it, trains the model, and writes:

- `models/cifar10_model.pkl`
- `models/metrics.json`

Training may take several minutes on a CPU.

### 3. Run the notebook

```powershell
jupyter notebook notebook/project_name.ipynb
```

Run every cell from top to bottom. The notebook includes acquisition, preprocessing, three feature interpretations, hyperparameter choices, training, accuracy, balanced accuracy, precision, recall, F1, confusion matrix, and prediction.

### 4. Start the API and UI

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

- UI: http://127.0.0.1:8000
- Swagger: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health
- Metrics: http://127.0.0.1:8000/metrics

## Docker

```powershell
docker compose up --build
```

Stop the containers:

```powershell
docker compose down
```

Build and run a container manually:

```powershell
docker build -t cifar10-mlops .
docker run --rm -p 8000:8000 -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models cifar10-mlops
```

## Prediction

Upload an image through the browser or Swagger UI. API example:

```powershell
curl.exe -F "file=@C:\path\to\image.jpg" http://127.0.0.1:8000/predict
```

## Retraining procedure

1. Upload multiple images in the browser and choose the correct label from `0` to `9`.
2. Images are saved into `data/uploads/<label>/` and recorded in `data/uploads.db`.
3. Press **Trigger retraining**.
4. The API calls `scripts/retrain.py` logic and uses `partial_fit` to update the existing classifier.
5. Check progress and history:

```powershell
curl.exe http://127.0.0.1:8000/retrain/status
```

For a ZIP, the ZIP contents should be placed in folders named `0` through `9`; the service assigns a supplied label to all extracted valid image types. Individual uploads use the selected label.

## API routes

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | Web UI |
| GET | `/health` | Uptime and model availability |
| POST | `/predict` | Predict one image |
| POST | `/upload` | Save multiple images or ZIP data |
| POST | `/retrain` | Trigger incremental retraining |
| GET | `/retrain/status` | Retraining state and history |
| GET | `/metrics` | Test metrics |
| GET | `/visualizations` | Chart data |

## Locust flood testing

Start the API, then run Locust:

```powershell
locust -f locustfile.py --host http://127.0.0.1:8000
```

Open http://127.0.0.1:8089, choose users and spawn rate, and run. Export the CSV report and HTML report.

Test one container, then two containers by running Locust against a load balancer or a second independent API instance. Record:

- Median and 95th-percentile latency
- Average response time
- Requests/second
- Failure percentage
- CPU and memory usage

Save reports under `results/locust/` and add screenshots to this README. Do not claim numbers until measured.

## Cloud deployment

The Docker image can be deployed to Render, Azure Container Apps, Google Cloud Run, or AWS ECS.

### Example: Render

1. Push this repository to GitHub.
2. Create a new Render Web Service.
3. Select Docker runtime.
4. Set the service port to `8000`.
5. Deploy and use the public URL.
6. Verify `/health`, `/docs`, and prediction.

For production persistence, mount object storage or persistent disk for `data/` and `models/`. The included SQLite database is suitable for demonstration and should be replaced with PostgreSQL for a production multi-replicated service.

## Tests

```powershell
python -m pip install pytest httpx
pytest -q
```

## Submission checklist

- [ ] Replace deployment placeholder with public URL
- [ ] Upload and link the camera-on YouTube demo
- [ ] Include the trained `.pkl` file or a clearly documented one-command training step
- [ ] Run and save notebook metrics
- [ ] Run Locust at 10, 50, and 100 users
- [ ] Save load-test screenshots and CSV/HTML reports
- [ ] Push the complete project to GitHub
- [ ] Submit the GitHub URL for attempt two
