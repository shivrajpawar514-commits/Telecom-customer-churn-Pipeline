import os
import sys
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)

from src.exception import CustomException
from src.logger import get_logger
from src.utils import load_object, save_json, load_numpy_array
from src import constants as C

logger = get_logger(__name__)


@dataclass
class ModelEvaluationConfig:
    model_path: str = C.MODEL_FILE_PATH
    metric_path: str = C.METRIC_FILE_PATH
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", C.MLFLOW_EXPERIMENT_NAME)
    registry_model_name: str = os.getenv("MODEL_REGISTRY_NAME", C.MODEL_REGISTRY_NAME)


class ModelEvaluation:
    """Computes evaluation metrics and versions/registers the model via MLflow's
    Model Registry, which serves as this pipeline's Model Versioning + Registry step."""

    def __init__(self, config: ModelEvaluationConfig = ModelEvaluationConfig()):
        self.config = config

    def initiate_model_evaluation(self, X_test, y_test, model_name: str = "model"):
        logger.info("Starting model evaluation")
        try:
            model = load_object(self.config.model_path)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred)),
                "recall": float(recall_score(y_test, y_pred)),
                "f1_score": float(f1_score(y_test, y_pred)),
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            }
            save_json(self.config.metric_path, metrics)
            logger.info(f"Evaluation metrics: {metrics}")

            # --- Model versioning + registry via MLflow ---
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)

            with mlflow.start_run(run_name=model_name):
                mlflow.log_params(model.get_params() if hasattr(model, "get_params") else {})
                mlflow.log_metrics({k: v for k, v in metrics.items() if k != "confusion_matrix"})

                # XGBoost estimators get their own MLflow flavor for robust
                # (de)serialization; everything else uses the sklearn flavor
                # with plain pickle serialization to avoid skops' untrusted-type
                # restrictions on non-sklearn-native estimators.
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

            return metrics

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    test_data = load_numpy_array(C.TRANSFORMED_TEST_FILE)
    ModelEvaluation().initiate_model_evaluation(test_data["X"], test_data["y"])
