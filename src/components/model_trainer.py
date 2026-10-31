import sys
from dataclasses import dataclass

from sklearn.dummy import DummyClassifier
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
        logger.info("Starting model training with candidate models and baselines")
        try:
            # Baseline and candidate model definitions
            # 1. Baseline: DummyClassifier (chance baseline)
            # 2. Baseline: Simple un-tuned LogisticRegression
            # 3. Candidate: Tuned LogisticRegression
            # 4. Candidate: Tuned RandomForest with class weighting
            # 5. Candidate: Tuned XGBoost with scale_pos_weight
            pos_ratio = (y_train == 0).sum() / max(1, (y_train == 1).sum())

            models = {
                "Dummy_Stratified": DummyClassifier(strategy="stratified", random_state=C.RANDOM_STATE),
                "Baseline_LogisticRegression": LogisticRegression(max_iter=1000, random_state=C.RANDOM_STATE),
                "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=C.RANDOM_STATE),
                "RandomForest": RandomForestClassifier(random_state=C.RANDOM_STATE, class_weight="balanced"),
                "XGBoost": XGBClassifier(
                    eval_metric="logloss",
                    scale_pos_weight=float(pos_ratio),
                    random_state=C.RANDOM_STATE,
                ),
            }

            param_grids = {
                "Dummy_Stratified": {},
                "Baseline_LogisticRegression": {},
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

            # Select champion among production candidates (exclude dummy and un-tuned baseline)
            candidates = ["LogisticRegression", "RandomForest", "XGBoost"]
            valid_candidates = {k: v for k, v in report.items() if k in candidates}
            best_name = max(valid_candidates, key=lambda k: valid_candidates[k]["roc_auc"])
            best_score = report[best_name]["roc_auc"]
            best_model = report[best_name]["model"]

            dummy_score = report.get("Dummy_Stratified", {}).get("roc_auc", 0.50)
            baseline_lr_score = report.get("Baseline_LogisticRegression", {}).get("roc_auc", 0.80)

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

            report_payload = {
                "best_model": best_name,
                "best_roc_auc": best_score,
                "best_cv_roc_auc_mean": report[best_name].get("cv_roc_auc_mean", best_score),
                "best_cv_roc_auc_std": report[best_name].get("cv_roc_auc_std", 0.0),
                "baseline_comparison": {
                    "dummy_stratified_roc_auc": dummy_score,
                    "baseline_untuned_lr_roc_auc": baseline_lr_score,
                    "champion_lift_over_dummy": round(best_score - dummy_score, 4),
                    "champion_lift_over_baseline_lr": round(best_score - baseline_lr_score, 4),
                },
                "all_models": {
                    k: {
                        "roc_auc": v["roc_auc"],
                        "cv_roc_auc_mean": v.get("cv_roc_auc_mean", v["roc_auc"]),
                        "cv_roc_auc_std": v.get("cv_roc_auc_std", 0.0),
                        "best_params": v["best_params"],
                    }
                    for k, v in report.items()
                },
            }
            save_json(self.config.report_path, report_payload)

            logger.info(
                f"Best model: {best_name} (Test ROC-AUC={best_score:.4f}, "
                f"Lift over Dummy={report_payload['baseline_comparison']['champion_lift_over_dummy']:.4f})"
            )
            return self.config.model_path, best_name, best_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    train_data = load_numpy_array(C.TRANSFORMED_TRAIN_FILE)
    test_data = load_numpy_array(C.TRANSFORMED_TEST_FILE)
    ModelTrainer().initiate_model_training(
        train_data["X"], train_data["y"], test_data["X"], test_data["y"]
    )
