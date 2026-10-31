import os

# --- Root paths ---
ARTIFACTS_DIR = "artifacts"
DATA_DIR = "data"

# --- Data ingestion ---
DEFAULT_DATA_SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
RAW_DATA_FILE = os.path.join(DATA_DIR, "raw", "telco_churn.csv")
INGESTED_TRAIN_FILE = os.path.join(ARTIFACTS_DIR, "train.csv")
INGESTED_TEST_FILE = os.path.join(ARTIFACTS_DIR, "test.csv")
DATASET_VERSION_PATH = os.path.join(ARTIFACTS_DIR, "dataset_version.json")
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COLUMN = "Churn"

# --- Data validation ---
SCHEMA_FILE_PATH = os.path.join("schema", "schema.yaml")
VALIDATION_REPORT_PATH = os.path.join(ARTIFACTS_DIR, "validation_report.json")

# --- Data transformation ---
PREPROCESSOR_OBJ_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
TRANSFORMED_TRAIN_FILE = os.path.join(ARTIFACTS_DIR, "train.npz")
TRANSFORMED_TEST_FILE = os.path.join(ARTIFACTS_DIR, "test.npz")

# --- Model training ---
MODEL_FILE_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
MODEL_REPORT_PATH = os.path.join(ARTIFACTS_DIR, "model_report.json")
EXPECTED_ROC_AUC_SCORE = 0.75

# --- Model evaluation / registry ---
METRIC_FILE_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
EXPERIMENT_MANIFEST_PATH = os.path.join(ARTIFACTS_DIR, "experiment_manifest.json")
MLFLOW_EXPERIMENT_NAME = "telecom-churn-prediction"
MODEL_REGISTRY_NAME = "telecom-churn-model"

# --- Monitoring & Alerting ---
MONITORING_DIR = os.path.join(ARTIFACTS_DIR, "monitoring")
DRIFT_ALERT_PATH = os.path.join(MONITORING_DIR, "drift_alert.json")
