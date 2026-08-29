# Telecom Customer Churn Prediction — MLOps Pipeline

An end-to-end MLOps pipeline for predicting telecom customer churn, from raw
data to a monitored production API on AWS.

```
Data Ingestion → Data Validation → Data Transformation → Model Training
   → Model Evaluation → Model Versioning → Model Registry
   → FastAPI Serving → Docker → CI/CD → AWS Deployment → Monitoring
```

Verified working end-to-end against the public **IBM Telco Customer Churn**
dataset (7,043 customers, 21 columns) — best model (XGBoost) scores
**ROC-AUC ≈ 0.845** on the held-out test set.

---

## Project structure

```
mlops-churn-pipeline/
├── data/
│   ├── raw/                     # downloaded source CSV
│   └── processed/                # (reserved for intermediate data)
├── notebooks/                    # EDA notebooks
├── schema/
│   └── schema.yaml               # expected columns, dtypes, target
├── src/
│   ├── components/
│   │   ├── data_ingestion.py     # download + train/test split
│   │   ├── data_validation.py    # schema + missing-value checks
│   │   ├── data_transformation.py# encoding/scaling pipeline
│   │   ├── model_trainer.py      # trains + selects best model
│   │   └── model_evaluation.py   # metrics + MLflow versioning/registry
│   ├── pipeline/
│   │   ├── training_pipeline.py  # orchestrates all components
│   │   └── prediction_pipeline.py# inference-time wrapper
│   ├── constants.py
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
├── monitoring/
│   └── monitor_drift.py          # Evidently data/target drift report
├── artifacts/                    # generated: models, preprocessors, reports
├── logs/                         # generated: run logs
├── app.py                        # FastAPI serving app
├── requirements.txt
├── Dockerfile
├── dvc.yaml                      # DVC pipeline stages (mirrors training_pipeline)
├── .github/workflows/main.yml    # CI/CD: build → ECR → ECS
└── tests/
```

---

## 1. Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in real values
```

Key `.env` variables:

| Variable | Purpose |
|---|---|
| `DATA_SOURCE_URL` | CSV URL to download on first run (defaults to the public Telco churn dataset) |
| `MLFLOW_TRACKING_URI` | Where MLflow logs runs/models (`sqlite:///mlflow.db` locally, a tracking server URL in prod) |
| `AWS_*`, `ECR_REPOSITORY`, `ECS_CLUSTER`, `ECS_SERVICE` | Used by the CI/CD workflow, not by the app itself |

## 2. Run the training pipeline

```bash
python -m src.pipeline.training_pipeline
```

This runs all six ML stages in order and leaves artifacts in `artifacts/`:
`train.csv`, `test.csv`, `validation_report.json`, `preprocessor.pkl`,
`train.npz`/`test.npz`, `model.pkl`, `model_report.json`, `metrics.json` —
and registers the model in the MLflow Model Registry
(`telecom-churn-model`, versioned automatically on every run).

Alternatively, run it as a reproducible **DVC** pipeline:
```bash
dvc repro
```

## 3. Serve predictions locally

```bash
uvicorn app:app --reload --port 8000
```

- `GET /health` — liveness check
- `POST /train` — triggers the training pipeline via the API
- `POST /predict` — returns `{"churn_prediction": "Yes"|"No", "churn_probability": 0.0-1.0}`

Example:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No",
    "tenure":12,"PhoneService":"Yes","MultipleLines":"No",
    "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"Yes",
    "DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"No",
    "Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
    "MonthlyCharges":70.35,"TotalCharges":845.5
  }'
```
Interactive docs at `http://localhost:8000/docs`.

## 4. Docker

```bash
docker build -t telecom-churn-pipeline:latest .
docker run -p 8000:8000 --env-file .env telecom-churn-pipeline:latest
```
Note: the container serves predictions from artifacts baked into the image
(or mounted via a volume) — run the training pipeline (locally or via
`POST /train`) before/after building so `artifacts/model.pkl` and
`artifacts/preprocessor.pkl` exist.

## 5. CI/CD (`.github/workflows/main.yml`)

- **CI** (every push/PR): install deps, `flake8` lint, `pytest` smoke tests.
- **CD** (push to `main` only): builds the Docker image, pushes to **Amazon
  ECR**, renders a new ECS task definition with the fresh image tag, and
  deploys it to an **ECS Fargate** service, waiting for stability.

Required GitHub repo secrets:
```
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION,
ECR_REPOSITORY, ECS_CLUSTER, ECS_SERVICE
```

This workflow assumes an ECS service/task definition named `churn-api`
already exists (see below) — the workflow only updates the image, it doesn't
provision infrastructure from scratch.

## 6. AWS deployment (one-time setup)

1. **ECR**: `aws ecr create-repository --repository-name telecom-churn-pipeline`
2. **ECS cluster**: `aws ecs create-cluster --cluster-name churn-cluster`
3. **Task definition**: register a Fargate task definition with container
   name `churn-api`, the ECR image, port `8000`, and a log configuration
   pointing at CloudWatch Logs.
4. **Service**: create an ECS service on that cluster, behind an
   **Application Load Balancer** with a target group health-checking
   `/health`, desired count ≥ 2 for availability.
5. **Secrets**: store `.env` values in **AWS Secrets Manager** or **SSM
   Parameter Store** and inject them into the task definition rather than
   baking them into the image.
6. Push to `main` — the CD job takes it from there.

## 7. Monitoring

- **App health**: `/health` is used by the Dockerfile `HEALTHCHECK` and the
  ALB target group.
- **Data/target drift**: `python monitoring/monitor_drift.py` uses Evidently
  to compare reference vs. current data and writes an HTML report to
  `artifacts/monitoring/`. Schedule this via EventBridge + Fargate/Lambda.
- **Infra metrics**: enable ECS Container Insights; alarm on CPU, memory,
  ALB 5xx rate, and p95 latency via CloudWatch Alarms → SNS.
- See `monitoring/README.md` for the full recommended setup.

## Model details

- **Target**: `Churn` (Yes/No → 1/0)
- **Candidates**: Logistic Regression, Random Forest, XGBoost — selected by
  ROC-AUC on a held-out test split, with `GridSearchCV` for hyperparameters.
- **Preprocessing**: median imputation + standard scaling for numeric
  features; most-frequent imputation + one-hot encoding for categoricals.
- **Versioning/Registry**: every training run is logged to MLflow
  (params, metrics, model artifact) and registered under
  `telecom-churn-model`, giving each run an incrementing version you can
  promote (`Staging` → `Production`) via the MLflow UI or client API.

## Extending this project

- Swap the `GridSearchCV` sweep in `model_trainer.py` for `Optuna`/`Hyperopt`
  if you need a larger search space.
- Add a `feature_store` component if features start being shared across
  models.
- Wire `monitor_drift.py`'s `dataset_drift` output into an automated
  retraining trigger (e.g., re-run `dvc repro` and re-deploy on drift).
