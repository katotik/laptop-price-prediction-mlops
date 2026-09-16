# Laptop Price Prediction MLOps

End-to-end MLOps project for the PMLDL Assignment 1 deployment task. The pipeline prepares raw laptop data, trains and evaluates a regression model, and deploys the model behind a FastAPI API plus a Streamlit web application in separate Docker containers.

## What Is Implemented

- Data engineering: loads `data/raw/laptops_Dataset.csv`, validates the input columns, removes duplicates and invalid prices, fills missing input values, and splits the data. Extreme prices are filtered from the training split only; the testing split keeps the original price range.
- Model engineering: parses laptop specification text into numeric features, one-hot encodes categorical fields, trains a `RandomForestRegressor`, saves `models/laptop_price_model.joblib`, and logs metrics to `models/metrics.json`.
- Deployment: FastAPI serves predictions at `/predict`; Streamlit provides input fields, a prediction button, and a prediction result area.
- Automation: `code/pipeline.py --watch --deploy --interval-seconds 300` runs the complete pipeline every 5 minutes.

## Dataset

The raw dataset is stored at `data/raw/laptops_Dataset.csv`.

Source: https://www.kaggle.com/datasets/waddahali/laptop-prices-and-specifications-dataset

License: CC BY-SA 4.0.

## Project Structure

```text
code/
  datasets/
    prepare_data.py
    inspect_dataset.py
  models/
    laptop_features.py
    train_model.py
  deployment/
    api/
      main.py
      Dockerfile
    app/
      streamlit_app.py
      Dockerfile
    docker-compose.yml
  pipeline.py
data/
  raw/
  processed/
models/
services/
  airflow/
requirements.txt
sample_request.json
```

## Local Setup

Use Python 3.11 for training before Docker deployment; the API image also uses Python 3.11. The local pipeline and tests work with Python 3.9 as well.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The Docker images install only the dependencies needed by their respective services.

## Run Pipeline Once

```bash
python3 code/pipeline.py
```

This creates:

- `data/processed/train.csv`
- `data/processed/test.csv`
- `models/laptop_price_model.joblib`
- `models/metrics.json`

## Run Tests

```bash
pip install -r requirements-dev.txt
python3 -m unittest discover -s tests -v
```

## Run API And App With Docker

Train the model first, then start both deployment containers:

```bash
python3 code/pipeline.py
docker compose -f code/deployment/docker-compose.yml up --build
```

Open:

- Streamlit app: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs
- API health: http://localhost:8000/health

Example API request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## Run Complete Automated Pipeline Every 5 Minutes

```bash
python3 code/pipeline.py --watch --deploy --interval-seconds 300
```

Each scheduled run performs data preparation, model training/evaluation, model packaging, and Docker Compose deployment. The interval is measured from the start of a run. Failed runs are logged and retried on the next interval. Keep this command running for the schedule to continue. If a run takes longer than 5 minutes on a slow machine, increase `--interval-seconds`.

## Notes For Demonstration

1. Start the automated pipeline with `python3 code/pipeline.py --watch --deploy --interval-seconds 300`.
2. Wait until Docker reports the API and app containers as running.
3. Open http://localhost:8501.
4. Fill in laptop fields and press `Predict price`.
5. Show that the prediction is returned by the API and displayed in the app.
