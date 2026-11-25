# Earthwise Butchery Image Classification MLOps Pipeline

An image-based classification pipeline for Earthwise Butchery. It predicts one of three broad product groups from a photo: **red meat**, **poultry**, or **fish**. The project uses non-tabular image data; class labels are used only as supervised-learning targets.

The provided source folder (`C:\Users\pc\Downloads\meat`) contains mixed, mostly unlabelled files. The label-review tool lets a person approve each image. Non-meat items, ambiguous images, screenshots, and mixed dishes should be skipped. Do not treat filename guesses as ground-truth labels.

## Current local prototype results

The initial visually reviewed selection contains 83 images: 44 red-meat, 30 poultry, and 9 fish. The deterministic split has 66 training images and 17 held-out test images. The current run measured 76.47% accuracy, 85.19% balanced accuracy, 77.78% macro-precision, 85.19% macro-recall, and 77.14% macro-F1. The test split contains just two fish examples, so these results are highly preliminary and must not be presented as reliable production performance. Confirm all provisional image labels and collect substantially more varied photos, especially fish, before drawing conclusions.

The classifier identifies only broad visual groups. It is not a food-safety, freshness, quality-grading, species-authentication, or traceability system.

## Pipeline and project structure

```text
app/main.py                 FastAPI endpoints and browser UI
data/labeled/<class>/       Human-reviewed images (local; not committed)
data/train/<class>/         Reproducible training split (generated; not committed)
data/test/<class>/          Held-out test split (generated; not committed)
models/                     Trained model, class names, and measured metrics
notebook/project_name.ipynb  Acquisition, preprocessing, insights, training, evaluation
scripts/label_images.py     Local image-review and labelling UI
scripts/prepare_dataset.py  Deterministic train/test split
scripts/train.py            Hyperparameter tuning and model training
scripts/retrain.py          Incremental retraining from labelled uploads
src/                         Shared preprocessing, model, and prediction code
tests/                       API and ML pipeline tests
locustfile.py                Load-test user
```

## 1. Install the project

```powershell
git clone https://github.com/kudakwashechikovo17/kudakwashe-mlops-classification-pipeline.git
cd kudakwashe-mlops-classification-pipeline
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.11 is also supported by the project dependencies.

## 2. Review and label the photos

Run the local review tool and pass the supplied image directory:

```powershell
python scripts/label_images.py "C:\Users\pc\Downloads\meat"
```

For each photo, choose **1: Red Meat**, **2: Poultry**, or **3: Fish**. Choose **0: Skip / not meat** for spices, eggs, screenshots, duplicates that are not exact copies, mixed dishes, or anything unclear. The tool records reviewed labels and copies approved photos into `data/labeled/<class>/`. It removes exact byte-identical duplicates from the review queue. Review labels carefully; do not label from filename alone.

Aim for at least 15–20 distinct images in each class if available. The split script requires at least five per class and the training script requires three training examples per class for cross-validation. More diverse, correctly labelled images improve the usefulness of evaluation.

## 3. Prepare the image dataset and train

```powershell
python scripts/prepare_dataset.py
python scripts/train.py
```

The split is reproducible (80% train, 20% test, fixed random seed). It keeps the held-out test images separate from hyperparameter selection. If you add more labels later, move or remove the generated `data/train/` and `data/test/` folders before preparing a new split.

Preprocessing applies EXIF orientation, RGB conversion, resizing, and normalization. Model features combine low-resolution RGB pixels and color histograms. Training uses a standardized logistic-loss SGD classifier with L2 regularization; three-fold stratified grid search selects the regularization strength using macro-F1. The model and measured metrics are written to:

- `models/earthwise_meat_model.pkl`
- `models/class_names.json`
- `models/earthwise_metrics.json`

Do not report results until `scripts/train.py` has run successfully on the human-labelled data.

## 4. Run the notebook

```powershell
jupyter notebook notebook/project_name.ipynb
```

Run all cells from top to bottom. The notebook explains acquisition/setup, preprocessing, three image-feature insights, hyperparameter tuning, at least five test metrics, a confusion matrix, and a single-image prediction compared with its known test label.

### Google Colab

1. Label the images locally as described above.
2. Create a ZIP containing the class folders directly:

   ```powershell
   Compress-Archive -Path .\data\labeled\* -DestinationPath .\earthwise_labeled.zip
   ```
3. Open the notebook from [GitHub in Colab](https://colab.research.google.com/github/kudakwashechikovo17/kudakwashe-mlops-classification-pipeline/blob/main/notebook/project_name.ipynb).
4. Select **Runtime → Run all**. When prompted, upload `earthwise_labeled.zip`.

The dataset is not stored in GitHub; upload it to your Colab runtime when prompted. Colab sessions are temporary, so download the executed notebook and model before the session ends.

## 5. Start the API and UI

After training:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open the UI at `http://127.0.0.1:8000`, Swagger at `/docs`, health at `/health`, metrics at `/metrics`, and image insights at `/visualizations`.

The UI supports single-image prediction, labelled bulk upload, and a retraining trigger. Uploaded images are saved in class-ID folders and recorded in SQLite before retraining. The saved fitted classifier is updated with `partial_fit`.

## 6. Docker and cloud deployment

```powershell
docker compose up --build
```

The container exposes port `8000`. Docker Compose mounts local `data/` and `models/` directories.

For a public Render deployment, push the repository to GitHub and create a new **Blueprint** from it. Render reads `render.yaml`, builds the Docker image, uses `/health` as its health check, and provisions persistent storage for uploaded files, SQLite, and updated model files. The initial trained model is copied into persistent model storage on first startup. A Render account and the configured paid persistent-disk plan are required; do not claim a public deployment until Render reports it is live and `/health` has been verified.

## 7. Locust load testing

Start the API, then in another terminal run:

```powershell
locust -f locustfile.py --host http://127.0.0.1:8000
```

Run tests with 10, 50, and 100 users. Record median and 95th-percentile latency, average response time, requests per second, failure percentage, and CPU/memory usage. Repeat with the required Docker container counts and save the actual Locust CSV/HTML reports and screenshots under `results/locust/`. Do not invent or estimate measured results.

### Local single-instance measurements

These are actual 20-second Locust runs against one local Uvicorn API process on this Windows workstation. They are **not** cloud results or multi-container comparisons. The Locust mix was four health checks per prediction request. CPU/memory was not captured.

| Concurrent users | Requests | Avg response | Median | 95th percentile | Requests/sec | Failures |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 481 | 44 ms | 5 ms | 18 ms | 29.04 | 0% |
| 50 | 2,755 | 21 ms | 12 ms | 81 ms | 152.65 | 0% |
| 100 | 4,945 | 68 ms | 28 ms | 240 ms | 251.76 | 0% |

Reports: [10 users](./results/locust/single-container-10-users.html), [50 users](./results/locust/single-container-50-users.html), [100 users](./results/locust/single-container-100-users.html). The corresponding CSV files are in the same folder. Run new tests against the deployed service and compare one versus two Docker containers before claiming the assignment's container-scaling requirement is complete.

## Tests

```powershell
python -m pytest -q
```

## Remaining submission work

- Verify the provisional labels on all 83 selected photos and collect substantially more images, especially fish; the current test set is too small to support robust performance claims.
- Deploy the Dockerized app from the Render Blueprint and add the verified public URL.
- Run Locust on one and two deployed Docker containers; capture CPU/memory and save the reports/screenshots.
- Record a camera-on YouTube demo of prediction and retraining.
- Add the demo link and final deployment/load-test results here.
- Submit the GitHub repository URL.
