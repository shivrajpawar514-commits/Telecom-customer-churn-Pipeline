import os
import shutil
import pytest
from fastapi.testclient import TestClient

from app import app
from src.pipeline.prediction_pipeline import PredictPipeline
from src.utils import load_object
from src import constants as C

client = TestClient(app)


@pytest.fixture
def sample_customer():
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 55.20,
        "TotalCharges": 1324.80,
    }


def test_model_loading_success():
    """Verify that model and preprocessor artifacts can be loaded successfully from disk."""
    assert os.path.exists(C.MODEL_FILE_PATH), "artifacts/model.pkl must exist"
    assert os.path.exists(C.PREPROCESSOR_OBJ_PATH), "artifacts/preprocessor.pkl must exist"

    model = load_object(C.MODEL_FILE_PATH)
    preprocessor = load_object(C.PREPROCESSOR_OBJ_PATH)

    assert model is not None
    assert preprocessor is not None
    assert hasattr(model, "predict")
    assert hasattr(preprocessor, "transform")


def test_model_loading_missing_file():
    """Verify that PredictPipeline raises FileNotFoundError when model path does not exist."""
    pipeline = PredictPipeline(
        model_path="artifacts/non_existent_model.pkl",
        preprocessor_path=C.PREPROCESSOR_OBJ_PATH,
    )
    with pytest.raises(Exception) as exc_info:
        pipeline._load()
    assert "not found" in str(exc_info.value).lower()


def test_health_check_endpoint():
    """Verify GET /health returns HTTP 200 and healthy status for container health checks."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_loaded" in data
    assert "service" in data


def test_container_healthcheck_compatibility():
    """Verify the endpoint responds properly to standard curl-style container probes."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_api_metrics_endpoint():
    """Verify GET /api/metrics returns structured pipeline metrics."""
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "metrics" in data
    assert "pipeline_status" in data
    assert "ge_validation" in data
    assert isinstance(data["ge_validation"], list)


def test_api_features_endpoint():
    """Verify GET /api/features extracts and ranks model feature importances."""
    resp = client.get("/api/features")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "features" in data
    assert len(data["features"]) > 0
    # Features should be sorted in descending order of importance
    importances = [f["importance"] for f in data["features"]]
    assert importances == sorted(importances, reverse=True)


def test_api_predict_all_models(sample_customer):
    """Verify POST /api/predict-all returns multi-model consensus predictions."""
    resp = client.post("/api/predict-all", json=sample_customer)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "predictions" in data
    assert "RandomForest" in data["predictions"] or "LogisticRegression" in data["predictions"]


def test_api_data_profile_endpoint():
    """Verify GET /api/data-profile dynamically computes summary statistics."""
    resp = client.get("/api/data-profile")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "num_stats" in data
    assert "cat_distributions" in data


def test_deployment_rollback(sample_customer):
    """Verify dynamic model promotion and deployment rollback mechanism.
    
    Workflow:
    1. Backup active champion model.
    2. Promote alternative model (e.g. LogisticRegression) -> verify active.
    3. Test inference works on promoted candidate.
    4. Trigger rollback by re-promoting original model -> verify restored.
    5. Test inference works on restored champion model.
    """
    backup_file = "artifacts/model_rollback_backup.pkl"
    shutil.copy(C.MODEL_FILE_PATH, backup_file)

    try:
        # Step 2: Promote alternative model
        promote_resp = client.post("/api/promote-model", json={"model_name": "LogisticRegression"})
        assert promote_resp.status_code == 200
        p_data = promote_resp.json()
        assert p_data["success"] is True

        # Step 3: Test prediction on new promoted model
        pred_resp1 = client.post("/predict", json=sample_customer)
        assert pred_resp1.status_code == 200
        assert pred_resp1.json()["success"] is True

        # Step 4: Simulate Rollback to previous model (RandomForest or original champion)
        rollback_resp = client.post("/api/promote-model", json={"model_name": "RandomForest"})
        assert rollback_resp.status_code == 200
        r_data = rollback_resp.json()
        assert r_data["success"] is True
        assert "RandomForest" in r_data["message"]

        # Step 5: Test prediction on restored model
        pred_resp2 = client.post("/predict", json=sample_customer)
        assert pred_resp2.status_code == 200
        assert pred_resp2.json()["success"] is True

    finally:
        # Restore backup
        if os.path.exists(backup_file):
            shutil.copy(backup_file, C.MODEL_FILE_PATH)
            os.remove(backup_file)
