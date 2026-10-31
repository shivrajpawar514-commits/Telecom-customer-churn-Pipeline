import os
import sys
import json
from datetime import datetime
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)

from src.exception import CustomException
from src.logger import get_logger
from src.utils import load_object, save_json, load_numpy_array, compute_file_hash
from src import constants as C

logger = get_logger(__name__)


@dataclass
class ModelEvaluationConfig:
    model_path: str = C.MODEL_FILE_PATH
    preprocessor_path: str = C.PREPROCESSOR_OBJ_PATH
    metric_path: str = C.METRIC_FILE_PATH
    model_report_path: str = C.MODEL_REPORT_PATH
    dataset_version_path: str = C.DATASET_VERSION_PATH
    experiment_manifest_path: str = C.EXPERIMENT_MANIFEST_PATH
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", C.MLFLOW_EXPERIMENT_NAME)
    registry_model_name: str = os.getenv("MODEL_REGISTRY_NAME", C.MODEL_REGISTRY_NAME)


class ModelEvaluation:
    """Computes evaluation metrics and versions/registers the model via MLflow's
    Model Registry and writes a comprehensive local Experiment Manifest."""

    def __init__(self, config: ModelEvaluationConfig = ModelEvaluationConfig()):
        self.config = config

    def initiate_model_evaluation(self, X_test, y_test, model_name: str = "model"):
        logger.info("Starting comprehensive model evaluation")
        try:
            model = load_object(self.config.model_path)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
                "pr_auc": float(average_precision_score(y_test, y_proba)),
                "log_loss": float(log_loss(y_test, y_proba)),
                "brier_score": float(brier_score_loss(y_test, y_proba)),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            }
            save_json(self.config.metric_path, metrics)
            logger.info(f"Evaluation metrics: {metrics}")

            # Load dataset metadata and trainer report if available
            dataset_meta = {}
            if os.path.exists(self.config.dataset_version_path):
                try:
                    with open(self.config.dataset_version_path, "r") as f:
                        dataset_meta = json.load(f)
                except Exception:
                    pass

            model_report = {}
            if os.path.exists(self.config.model_report_path):
                try:
                    with open(self.config.model_report_path, "r") as f:
                        model_report = json.load(f)
                except Exception:
                    pass

            # Create comprehensive experiment manifest
            manifest = {
                "manifest_version": "1.0",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "champion_model": model_name,
                "dataset_version": dataset_meta,
                "evaluation_metrics": metrics,
                "baseline_comparison": model_report.get("baseline_comparison", {}),
                "model_parameters": model.get_params() if hasattr(model, "get_params") else {},
                "artifacts": {
                    "model_file": self.config.model_path,
                    "model_sha256": compute_file_hash(self.config.model_path),
                    "preprocessor_file": self.config.preprocessor_path,
                    "preprocessor_sha256": compute_file_hash(self.config.preprocessor_path),
                    "metrics_file": self.config.metric_path,
                    "model_report_file": self.config.model_report_path,
                },
                "split_strategy": {
                    "method": "Stratified 80/20 train/test split",
                    "random_state": C.RANDOM_STATE,
                    "test_size": C.TEST_SIZE,
                    "cross_validation": "3-Fold Stratified K-Fold CV during model selection",
                },
            }
            save_json(self.config.experiment_manifest_path, manifest)
            logger.info(f"Experiment manifest saved to {self.config.experiment_manifest_path}")

            # --- Model versioning + registry via MLflow ---
            try:
                mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
                mlflow.set_experiment(self.config.experiment_name)

                with mlflow.start_run(run_name=model_name):
                    # Log parameters
                    mlflow.log_params(model.get_params() if hasattr(model, "get_params") else {})
                    mlflow.log_param("champion_model", model_name)
                    if "raw_data_sha256" in dataset_meta:
                        mlflow.log_param("dataset_sha256", dataset_meta["raw_data_sha256"])

                    # Log metrics
                    mlflow.log_metrics({k: v for k, v in metrics.items() if k != "confusion_matrix"})

                    # XGBoost estimators get their own MLflow flavor; everything else uses sklearn
                    if isinstance(model, XGBClassifier):
                        mlflow.xgboost.log_model(
                            xgb_model=model,
                            artifact_path="model",
                            registered_model_name=self.config.registry_model_name,
                        )
                    else:
                        mlflow.sklearn.log_model(
                            sk_model=model,
                            artifact_path="model",
                            registered_model_name=self.config.registry_model_name,
                            serialization_format="pickle",
                        )
                    logger.info(
                        f"Model registered to MLflow registry as '{self.config.registry_model_name}'"
                    )
            except Exception as mlf_err:
                logger.warning(
                    f"MLflow tracking/registration encountered an issue: {mlf_err}. "
                    "Local artifact and manifest tracking completed successfully."
                )

            return metrics

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    test_data = load_numpy_array(C.TRANSFORMED_TEST_FILE)
    ModelEvaluation().initiate_model_evaluation(test_data["X"], test_data["y"])
