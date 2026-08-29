import os
import sys
import json
import yaml
import dill
import numpy as np

from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)


def save_object(file_path: str, obj) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
        logger.info(f"Saved object to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    try:
        with open(file_path, "rb") as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e, sys)


def save_json(file_path: str, data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4, default=str)
    except Exception as e:
        raise CustomException(e, sys)


def read_yaml(file_path: str) -> dict:
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise CustomException(e, sys)


def save_numpy_array(file_path: str, **arrays) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savez(file_path, **arrays)
    except Exception as e:
        raise CustomException(e, sys)


def load_numpy_array(file_path: str):
    try:
        return np.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(models: dict, param_grids: dict, X_train, y_train, X_test, y_test) -> dict:
    """Fit each candidate model (optionally with GridSearchCV) and score on test set.
    Returns {model_name: {"model": fitted_estimator, "roc_auc": float, "best_params": dict}}
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import roc_auc_score

    report = {}
    try:
        for name, model in models.items():
            params = param_grids.get(name, {})
            if params:
                gs = GridSearchCV(model, params, cv=3, scoring="roc_auc", n_jobs=-1)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
                best_params = gs.best_params_
            else:
                model.fit(X_train, y_train)
                best_model = model
                best_params = {}

            y_proba = best_model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_proba)

            report[name] = {
                "model": best_model,
                "roc_auc": float(score),
                "best_params": best_params,
            }
            logger.info(f"{name}: ROC-AUC={score:.4f} params={best_params}")

        return report
    except Exception as e:
        raise CustomException(e, sys)
