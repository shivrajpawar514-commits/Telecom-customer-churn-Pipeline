import os
import sys
import json
import yaml
import hashlib
import pickle
import numpy as np

try:
    import joblib
except ImportError:
    joblib = None

try:
    import dill
except ImportError:
    dill = None

from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file for artifact and dataset versioning."""
    if not os.path.exists(file_path):
        return ""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_object(file_path: str, obj) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if joblib is not None:
            joblib.dump(obj, file_path)
        elif dill is not None:
            with open(file_path, "wb") as f:
                dill.dump(obj, f)
        else:
            with open(file_path, "wb") as f:
                pickle.dump(obj, f)
        logger.info(f"Saved object to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    try:
        if joblib is not None:
            try:
                return joblib.load(file_path)
            except Exception:
                pass
        if dill is not None:
            try:
                with open(file_path, "rb") as f:
                    return dill.load(f)
            except Exception:
                pass
        with open(file_path, "rb") as f:
            return pickle.load(f)
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
    Returns {model_name: {"model": fitted_estimator, "roc_auc": float, "cv_roc_auc_mean": float, "best_params": dict}}
    """
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score

    report = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    try:
        for name, model in models.items():
            params = param_grids.get(name, {})
            if params:
                gs = GridSearchCV(model, params, cv=cv, scoring="roc_auc", n_jobs=-1)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
                best_params = gs.best_params_
                cv_mean = float(gs.best_score_)
                cv_std = float(gs.cv_results_["std_test_score"][gs.best_index_])
            else:
                model.fit(X_train, y_train)
                best_model = model
                best_params = {}
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
                cv_mean = float(np.mean(cv_scores))
                cv_std = float(np.std(cv_scores))

            y_proba = best_model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_proba)

            report[name] = {
                "model": best_model,
                "roc_auc": float(score),
                "cv_roc_auc_mean": cv_mean,
                "cv_roc_auc_std": cv_std,
                "best_params": best_params,
            }
            logger.info(
                f"{name}: Test ROC-AUC={score:.4f} (CV ROC-AUC={cv_mean:.4f} +/- {cv_std:.4f}) params={best_params}"
            )

        return report
    except Exception as e:
        raise CustomException(e, sys)
