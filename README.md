# Telecom Customer Churn Prediction — Production MLOps Pipeline

[![CI/CD Pipeline](https://github.com/shivrajpawar514-commits/Telecom-customer-churn-Pipeline/actions/workflows/main.yml/badge.svg)](https://github.com/shivrajpawar514-commits/Telecom-customer-churn-Pipeline/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, end-to-end MLOps pipeline for predicting telecom customer churn, engineered to demonstrate **reproducibility, rigorous testing, continuous observability, and safe zero-downtime deployment**.

```
Data Ingestion ──► Data Validation ──► Data Transformation ──► Model Training
       │                   │                     │                     │
       ▼                   ▼                     ▼                     ▼
 Dataset Version     Schema Assertions     Preprocessor Fit      Candidate Benchmarking
  (SHA-256 Hashes)   Missing Thresholds    OneHot / Scaler       & Cross-Validation
                                                                       │
                                                                       ▼
   Monitoring & Alerts ◄── Deployment & Serving ◄── Model Evaluation & Manifest
   (Evidently Drift,       (FastAPI, Rollback,       (Held-out ROC-AUC vs Baselines,
    Human Review SOP)       Health Probes)            Experiment Lineage)
```

---

## Highlights & Core MLOps Guarantees

- **One-Command Reproducibility:** Clean-environment setup and full pipeline execution in a single command (`./reproduce.sh` or `make reproduce`).
- **Comprehensive Quality Assurance (32+ Tests):** Automated tests covering schema mutations, missing value tolerance, unseen categorical levels, extreme class imbalance, boundary validation on predictions, model loading, and container health checks.
- **Consistent Experiment & Artifact Tracking:** Cryptographic SHA-256 recording for dataset versions, model binaries, fitted hyperparameters, and evaluation manifests (`artifacts/experiment_manifest.json`).
- **Rigorous ROC-AUC Generation & Baseline Benchmarking:** Documented Stratified 80/20 split strategy with 3-Fold Stratified Cross-Validation, benchmarked against a Dummy Prior Baseline and an Untuned Logistic Regression baseline.
- **Governed Drift Alerting (Zero Automated Retraining):** Multi-tier drift detection using Evidently AI, explicitly decoupled from auto-retraining to protect against label lag, data corruption, and silent production degradation.
- **Deployment Safety & Instant Rollback:** Liveness/readiness health probes (`/health`), safe cloud credential injection via AWS Secrets Manager/SSM, and hot-swappable model rollback (`/api/promote-model`).

---

## Project Structure

```
Telecom-customer-churn-Pipeline/
├── reproduce.sh                 # One-command full reproducibility runner
├── Makefile                     # Standardized developer workflows (setup, train, test, drift)
├── requirements.txt             # Pinned production and testing dependencies
├── Dockerfile                   # Multi-stage container definition with HEALTHCHECK
├── app.py                       # FastAPI application (serving, monitoring, rollback)
├── dvc.yaml                     # DVC pipeline stages mirroring the Python pipeline
├── data/
│   ├── raw/                     # Ingested dataset (telco_churn.csv)
│   └── processed/               # Processed splits
├── schema/
│   └── schema.yaml              # Schema specification (columns, dtypes, constraints)
├── src/
│   ├── components/
│   │   ├── data_ingestion.py    # Download, synthetic fallback, SHA-256 versioning
│   │   ├── data_validation.py   # Schema assertions, null checks, imbalance checks
│   │   ├── data_transformation.py # Sklearn pipeline, unseen category handling
│   │   ├── model_trainer.py     # Stratified CV, GridSearch, baseline models
│   │   └── model_evaluation.py  # Held-out ROC-AUC, lift calculation, manifest generation
│   ├── pipeline/
│   │   ├── training_pipeline.py # Orchestrates all 6 pipeline stages
│   │   └── prediction_pipeline.py # Production inference engine with safety checks
│   ├── constants.py             # System paths and configuration constants
│   ├── logger.py                # Rotating structured file/console logger
│   ├── exception.py             # Detailed traceback exception handler
│   └── utils.py                 # Cryptographic hashing, robust model serialization
├── monitoring/
│   ├── monitor_drift.py         # Evidently data/target drift auditor
│   └── README.md                # Drift monitoring architecture & operational guidelines
├── docs/
│   ├── ROC_AUC_METHODOLOGY.md   # Split strategy, CV design, baseline comparisons
│   ├── DRIFT_RESPONSE_AND_ALERTING.md # Severity tiers, alert schema, human review SOP
│   ├── SECRETS_MANAGEMENT_AND_DEPLOYMENT_SAFETY.md # Cloud credentials & container safety
│   ├── PROJECT_REPORT.md        # Academic/technical report
│   └── PRESENTATION.md          # System presentation slides
├── tests/
│   ├── test_data_validation.py  # Schema, missing values, unseen categories, imbalance
│   ├── test_prediction_validation.py # Pydantic boundary checks & missing field tests
│   ├── test_deployment_and_serving.py # Model loading, /health, rollback, features
│   ├── test_drift_alerting.py   # Drift tier evaluation and policy enforcement
│   ├── test_secrets_safety.py   # Scanner for hardcoded credentials & gitignore rules
│   └── test_pipeline_smoke.py   # Module import sanity checks
└── artifacts/                   # Cryptographically versioned manifests and models
```

---

## 1. Quick Start & One-Command Reproducibility

### The One-Command Runner
Execute the complete pipeline from scratch—environment verification, ingestion, transformation, candidate model training, held-out evaluation, full test suite, and drift detection—with a single command:

```bash
./reproduce.sh
```
Or via `make`:
```bash
make reproduce
```

### Clean Environment Setup (Manual)
To set up a fresh virtual environment manually:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

### Individual Execution Targets
```bash
make train      # Runs python -m src.pipeline.training_pipeline
make test       # Runs pytest -v tests/ (32 tests)
make drift      # Runs python monitoring/monitor_drift.py
make serve      # Launches FastAPI on http://localhost:8000
```

---

## 2. Automated Test Suite

The repository contains 32 automated tests verifying production reliability:

| Test Module | Coverage Focus | Key Assertions |
|---|---|---|
| `test_data_validation.py` | Schema & Ingestion Quality | Rejects missing columns, unexpected columns, and dtype mismatches. Passes null rates $< 30\%$ and fails $> 30\%$. Tests OneHotEncoder handling of unseen categories (`handle_unknown='ignore'`). Detects extreme class imbalance. |
| `test_prediction_validation.py` | API Input Guardrails | Validates strict Pydantic payload parsing. Rejects omitted fields, non-numeric strings, negative tenure, negative charges, and out-of-bounds `SeniorCitizen` flags with HTTP 422. |
| `test_deployment_and_serving.py` | Serving & Resilience | Verifies successful model loading, graceful handling of missing files, HTTP 200 `/health` probes, multi-model consensus prediction, and instant deployment rollback via `/api/promote-model`. |
| `test_drift_alerting.py` | Observability & Policy | Verifies calculation of drift severity tiers (`NO_DRIFT`, `INFO`, `WARNING`, `CRITICAL`), HTML report generation, and asserts that automated retraining and auto-deploy flags are set to `False`. |
| `test_secrets_safety.py` | Security & Compliance | Scans source files for regex `AKIA...` AWS access keys, verifies `.env` is unindexed in git, ensures `.gitignore` and `.dockerignore` mask sensitive files, and audits `.env.example` placeholders. |

Run tests at any time with:
```bash
pytest -v tests/
```

---

## 3. Experiment & Artifact Tracking

Every execution of the pipeline writes cryptographically validated audit trails to `artifacts/`:

1. **Dataset Versioning (`artifacts/dataset_version.json`):**
   - SHA-256 hashes of the raw download, training partition (`train.csv`), and held-out partition (`test.csv`).
   - Split ratio, record counts ($800$ train / $200$ test), random seed ($42$), and target class distributions.
2. **Model Reports (`artifacts/model_report.json`):**
   - 3-Fold Cross-Validation mean ROC-AUC and fold standard deviation for all candidate architectures.
   - Best hyperparameter combinations found via `GridSearchCV`.
3. **Experiment Manifest (`artifacts/experiment_manifest.json`):**
   - Comprehensive record linking dataset version, champion model parameters, baseline comparisons, metrics, and SHA-256 hashes of the serialized `model.pkl` and `preprocessor.pkl`.
4. **Metrics Manifest (`artifacts/metrics.json`):**
   - Held-out evaluation metrics: ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1-Score, Log-Loss, Brier Score, and Confusion Matrix.

---

## 4. ROC-AUC Generation & Baseline Comparison

For detailed documentation, see [`docs/ROC_AUC_METHODOLOGY.md`](docs/ROC_AUC_METHODOLOGY.md).

### Split Strategy
- **Stratified Holdout Split:** 80% train / 20% test using `stratify=y` with seed `42` to prevent class distribution distortion.
- **Cross-Validation:** 3-Fold Stratified K-Fold CV during model selection to prevent fold variance and overfitting.
- **Zero-Leakage Guarantee:** Imputers and encoders are fitted **strictly on the training partition**.

### Baseline Comparison Table
To prove real business efficacy, the champion model is explicitly compared against two baselines:

| Model Architecture | Role | 3-Fold CV ROC-AUC | Held-out Test ROC-AUC | Lift over Dummy Baseline |
|---|---|---|---|---|
| **Stratified Dummy** | Prior Baseline | $0.4776 \pm 0.0089$ | **0.4972** | Baseline ($0.0000$) |
| **Untuned Logistic Reg.** | Linear Baseline | $0.9836 \pm 0.0084$ | **0.9941** | $+0.4969$ |
| **Tuned Logistic Reg.** | Candidate | $0.9918 \pm 0.0045$ | **0.9964** | $+0.4992$ |
| **Random Forest (Champion)**| Candidate | $1.0000 \pm 0.0000$ | **1.0000** | **+0.5028** |
| **XGBoost** | Candidate | $1.0000 \pm 0.0000$ | **1.0000** | **+0.5028** |

---

## 5. Drift Detection & Governance Review Process

For detailed documentation, see [`docs/DRIFT_RESPONSE_AND_ALERTING.md`](docs/DRIFT_RESPONSE_AND_ALERTING.md).

### Safety Policy: No Automated Retraining on Drift
> [!CAUTION]
> **Automated retraining and automated re-deployment upon drift detection are strictly prohibited.**
>
> 1. **Label Delay:** Ground truth churn takes **30–60 days** to mature. Retraining on live inference data before labels mature forces the pipeline to train on unverified pseudo-labels.
> 2. **Telemetry Protection:** Upstream sensor defects, tracking script errors, or schema mutations must not be allowed to poison the training distribution.
> 3. **Transient Outliers:** Promotional campaigns and holiday spikes represent temporary shifts that should not displace a verified champion model.

### Multi-Tier Alerting & Review SOP
1. **Evidently Drift Check:** `python monitoring/monitor_drift.py` computes Wasserstein/K-S drift for numeric features and Chi-Square/J-S divergence for categoricals.
2. **Alert Manifest:** Generates `artifacts/monitoring/drift_alert.json` classifying status into `NO_DRIFT`, `INFO`, `WARNING`, or `CRITICAL`.
3. **5-Phase Human-in-the-Loop SOP:**
   - **Phase 1: Triage:** Inspect `drift_report_*.html` p-values and visual empirical distributions.
   - **Phase 2: Upstream Audit:** Verify ETL pipelines, API logs, and schema integrity to rule out telemetry bugs.
   - **Phase 3: Label Maturity Audit:** Ensure 60-day matured billing outcomes are available before considering retraining.
   - **Phase 4: Offline Shadow Training:** Retrain candidate models in an isolated feature branch and benchmark against champion and baselines.
   - **Phase 5: Governance Sign-off & Canary Rollout:** Model Governance Committee review followed by a staged canary deployment (10% traffic) with instant rollback capabilities.

---

## 6. Container Health Checks & Deployment Rollback

### Health Checks & Liveness Probes
The container exposes `GET /health` which validates both process liveness and model readiness:
```json
{
  "status": "healthy",
  "service": "telecom-churn-api",
  "version": "1.5.0",
  "model_loaded": true,
  "preprocessor_loaded": true
}
```

The Dockerfile configures an active container probe:
```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

### Instant Deployment Rollback
If an operational regression occurs in production:
1. Candidate models evaluated during training (`model_LogisticRegression.pkl`, `model_RandomForest.pkl`, `model_XGBoost.pkl`) remain staged in `artifacts/`.
2. Operators can hot-swap the active model in memory via `POST /api/promote-model`:
   ```bash
   curl -X POST http://localhost:8000/api/promote-model \
     -H "Content-Type: application/json" \
     -d '{"model_name": "RandomForest"}'
   ```
3. Verified in automated test `test_deployment_rollback`.

---

## 7. Cloud Secrets & Credentials Security

For detailed documentation, see [`docs/SECRETS_MANAGEMENT_AND_DEPLOYMENT_SAFETY.md`](docs/SECRETS_MANAGEMENT_AND_DEPLOYMENT_SAFETY.md).

- **Zero Hardcoded Credentials:** Verified by `test_no_hardcoded_aws_keys_in_source`.
- **Environment Separation:** Local development uses an uncommitted `.env` file (strictly excluded via `.gitignore` and `.dockerignore`).
- **Production Secrets Injection:** AWS ECS tasks ingest credentials at runtime via the task definition `secrets` block referencing **AWS Secrets Manager** ARNs or **SSM Parameter Store** parameters. No secret keys or certificates are ever baked into Docker image layers.

---

## 8. Prediction API Reference

Start the local server:
```bash
uvicorn app:app --reload --port 8000
```

### Sample Prediction Request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 845.5
  }'
```

### Sample Response:
```json
{
  "success": true,
  "prediction": {
    "churn_prediction": "Yes",
    "churn_probability": 0.8125
  },
  "explanations": {
    "risk_level": "High Risk",
    "primary_drivers": ["Contract", "tenure", "MonthlyCharges"]
  }
}
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

---

## 9. Verification & Audit Checklist

- [x] Complete pipeline reproducible via `./reproduce.sh` and `make reproduce`.
- [x] Automated tests for schema changes, missing values, unseen categories, and imbalanced classes.
- [x] Input validation with Pydantic boundary checks and HTTP 422 error reporting.
- [x] Dataset versioning with SHA-256 hashes, parameters, metrics, and artifact manifest.
- [x] Documented ROC-AUC methodology, stratified split strategy, and dual-baseline comparisons.
- [x] Multi-tier drift detection with Evidently, alert manifest, and strict human review SOP (no auto-retrain).
- [x] Tested model loading, `/health` liveness checks, and hot-swap deployment rollback.
- [x] Cloud credentials security audited via automated tests and runtime secret injection architecture.
