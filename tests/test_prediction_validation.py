import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


@pytest.fixture
def valid_customer_payload():
    return {
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
        "TotalCharges": 845.5,
    }


def test_predict_valid_payload(valid_customer_payload):
    """Ensure valid customer records produce a 200 OK response with prediction schema."""
    resp = client.post("/predict", json=valid_customer_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "prediction" in data
    assert data["prediction"]["churn_prediction"] in ["Yes", "No"]
    assert 0.0 <= data["prediction"]["churn_probability"] <= 1.0
    assert "explanations" in data


def test_predict_missing_mandatory_field(valid_customer_payload):
    """Ensure omitting a required field triggers HTTP 422 Unprocessable Entity."""
    payload = valid_customer_payload.copy()
    del payload["Contract"]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    errors = resp.json()["detail"]
    assert any("Contract" in err["loc"] for err in errors)


def test_predict_invalid_datatype(valid_customer_payload):
    """Ensure invalid data types (e.g. string for MonthlyCharges) trigger HTTP 422."""
    payload = valid_customer_payload.copy()
    payload["MonthlyCharges"] = "seventy-dollars"
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    errors = resp.json()["detail"]
    assert any("MonthlyCharges" in err["loc"] for err in errors)


def test_predict_negative_tenure_validation(valid_customer_payload):
    """Ensure negative tenure triggers validation error (422)."""
    payload = valid_customer_payload.copy()
    payload["tenure"] = -5
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_negative_monthly_charges_validation(valid_customer_payload):
    """Ensure negative monthly charges trigger validation error (422)."""
    payload = valid_customer_payload.copy()
    payload["MonthlyCharges"] = -15.50
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_senior_citizen_bounds(valid_customer_payload):
    """Ensure SeniorCitizen only accepts 0 or 1."""
    payload = valid_customer_payload.copy()
    payload["SeniorCitizen"] = 5
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_empty_payload():
    """Ensure completely empty request body triggers 422."""
    resp = client.post("/predict", json={})
    assert resp.status_code == 422
