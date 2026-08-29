import sys

from src.exception import CustomException
from src.logger import get_logger

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

logger = get_logger(__name__)


def run_training_pipeline():
    try:
        logger.info("=== TRAINING PIPELINE START ===")

        train_path, test_path = DataIngestion().initiate_data_ingestion()

        DataValidation().initiate_data_validation(train_path, test_path)

        X_train, y_train, X_test, y_test, _ = DataTransformation().initiate_data_transformation(
            train_path, test_path
        )

        model_path, best_name, best_score = ModelTrainer().initiate_model_training(
            X_train, y_train, X_test, y_test
        )

        metrics = ModelEvaluation().initiate_model_evaluation(X_test, y_test, model_name=best_name)

        logger.info(f"=== TRAINING PIPELINE COMPLETE: {best_name} -> {metrics} ===")
        return {"best_model": best_name, "train_roc_auc": best_score, "test_metrics": metrics}

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
