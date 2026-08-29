import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_object, save_json, evaluate_models, load_numpy_array
from src import constants as C

logger = get_logger(__name__)


@dataclass
class ModelTrainerConfig:
    model_path: str = C.MODEL_FILE_PATH
    report_path: str = C.MODEL_REPORT_PATH
    expected_roc_auc: float = C.EXPECTED_ROC_AUC_SCORE


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config

    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        logger.info("Starting model training")
        try:
            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "RandomForest": RandomForestClassifier(random_state=C.RANDOM_STATE),
                "XGBoost": XGBClassifier(
                    eval_metric="logloss", random_state=C.RANDOM_STATE
                ),
            }

            param_grids = {
                "LogisticRegression": {"C": [0.1, 1.0, 10.0]},
                "RandomForest": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, None],
                },
                "XGBoost": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5],
                    "learning_rate": [0.05, 0.1],
                },
            }

            report = evaluate_models(models, param_grids, X_train, y_train, X_test, y_test)

            best_name = max(report, key=lambda k: report[k]["roc_auc"])
            best_score = report[best_name]["roc_auc"]
            best_model = report[best_name]["model"]

            if best_score < self.config.expected_roc_auc:
                logger.warning(
                    f"Best model {best_name} scored {best_score:.4f}, "
                    f"below expected {self.config.expected_roc_auc}"
                )

            save_object(self.config.model_path, best_model)
            
            # Save candidate models for sandbox registry promotions
            import os
            for name, details in report.items():
                m_obj = details["model"]
                save_object(os.path.join(os.path.dirname(self.config.model_path), f"model_{name}.pkl"), m_obj)

            save_json(
                self.config.report_path,
                {
                    "best_model": best_name,
                    "best_roc_auc": best_score,
                    "all_models": {
                        k: {"roc_auc": v["roc_auc"], "best_params": v["best_params"]}
                        for k, v in report.items()
                    },
                },
            )

            logger.info(f"Best model: {best_name} (ROC-AUC={best_score:.4f})")
            return self.config.model_path, best_name, best_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    train_data = load_numpy_array(C.TRANSFORMED_TRAIN_FILE)
    test_data = load_numpy_array(C.TRANSFORMED_TEST_FILE)
    ModelTrainer().initiate_model_training(
        train_data["X"], train_data["y"], test_data["X"], test_data["y"]
    )
