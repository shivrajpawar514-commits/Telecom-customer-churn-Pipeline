import numpy as np
import pandas as pd
import pytest

from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData
from src import constants as C


@pytest.fixture
def sample_valid_df():
    """Generates a valid dataframe conforming to schema/schema.yaml."""
    from src.components.data_ingestion import generate_synthetic_telco_churn
    return generate_synthetic_telco_churn(n_samples=100, random_state=42)


def test_schema_valid_dataframe(sample_valid_df):
    """Ensure a clean, properly formatted dataset passes validation."""
    validator = DataValidation()
    res = validator.validate_dataframe(sample_valid_df)
    assert res["passed"] is True
    assert len(res["columns"]["missing_columns"]) == 0
    assert len(res["dtypes"]["mismatches"]) == 0


def test_schema_missing_columns(sample_valid_df):
    """Ensure missing columns are caught and cause validation to fail."""
    validator = DataValidation()
    corrupt_df = sample_valid_df.drop(columns=["Contract", "tenure"])
    res = validator.validate_dataframe(corrupt_df)
    assert res["passed"] is False
    assert "Contract" in res["columns"]["missing_columns"]
    assert "tenure" in res["columns"]["missing_columns"]


def test_schema_unexpected_columns(sample_valid_df):
    """Ensure unexpected new columns are identified in validation reports."""
    validator = DataValidation()
    extra_df = sample_valid_df.copy()
    extra_df["unregistered_feature_1"] = 123
    extra_df["unregistered_feature_2"] = "unknown"
    res = validator.validate_dataframe(extra_df)
    assert "unregistered_feature_1" in res["columns"]["unexpected_columns"]
    assert "unregistered_feature_2" in res["columns"]["unexpected_columns"]


def test_schema_dtype_mismatch(sample_valid_df):
    """Ensure column dtype mismatches (e.g. tenure passed as text) are caught."""
    validator = DataValidation()
    mismatch_df = sample_valid_df.copy()
    mismatch_df["tenure"] = mismatch_df["tenure"].astype(str) + "_months"
    res = validator.validate_dataframe(mismatch_df)
    assert res["passed"] is False
    assert "tenure" in res["dtypes"]["mismatches"]


def test_missing_values_within_threshold(sample_valid_df):
    """Verify that acceptable missing values (e.g. TotalCharges nulls < 30%) pass validation."""
    validator = DataValidation()
    df = sample_valid_df.copy()
    df.loc[:5, "TotalCharges"] = np.nan  # 6% missing, well below 30% threshold
    res = validator.validate_dataframe(df)
    assert res["passed"] is True
    assert len(res["missing_values"]["offending_columns"]) == 0


def test_missing_values_exceeding_threshold(sample_valid_df):
    """Verify that excessive missing values (> 30%) fail validation."""
    validator = DataValidation()
    df = sample_valid_df.copy()
    df.loc[:40, "MonthlyCharges"] = np.nan  # 41% missing, exceeds 30% threshold
    res = validator.validate_dataframe(df)
    assert res["passed"] is False
    assert "MonthlyCharges" in res["missing_values"]["offending_columns"]


def test_unseen_categories_handling(sample_valid_df):
    """Verify that unseen categorical values do not crash transformation or prediction."""
    transformer = DataTransformation()
    preprocessor = transformer.get_preprocessor()

    drop_cols = ["customerID", "Churn"]
    X = sample_valid_df.drop(columns=[c for c in drop_cols if c in sample_valid_df.columns])
    preprocessor.fit(X)

    # Introduce unseen categories never present in training data
    novel_df = X.iloc[:5].copy()
    novel_df["Contract"] = "Decade-to-decade"
    novel_df["PaymentMethod"] = "Cryptocurrency"
    novel_df["InternetService"] = "Quantum-Broadband"

    # OneHotEncoder(handle_unknown='ignore') must ignore these and output zeros
    transformed = preprocessor.transform(novel_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    assert transformed.shape[0] == 5
    assert not np.isnan(transformed).any()

    # Verify that prediction pipeline with trained artifacts handles unseen categories
    predictor = PredictPipeline()
    pred_res = predictor.predict(novel_df.iloc[[0]])
    assert "churn_prediction" in pred_res
    assert pred_res["churn_prediction"] in ["Yes", "No"]
    assert 0.0 <= pred_res["churn_probability"] <= 1.0


def test_imbalanced_classes_detection():
    """Verify class balance validation checks for both healthy and severe imbalance."""
    validator = DataValidation()

    # Normal imbalance (e.g. 75/25 churn ratio) should pass
    normal_df = pd.DataFrame({"Churn": ["No"] * 75 + ["Yes"] * 25})
    res_normal = validator._validate_class_imbalance(normal_df)
    assert res_normal["passed"] is True
    assert 0.20 <= res_normal["positive_class_ratio"] <= 0.30

    # Severe/extreme imbalance (99% No, 1% Yes) should flag a warning/failure
    extreme_df = pd.DataFrame({"Churn": ["No"] * 99 + ["Yes"] * 1})
    res_extreme = validator._validate_class_imbalance(extreme_df)
    assert res_extreme["passed"] is False
    assert res_extreme["positive_class_ratio"] == 0.01

    # Single-class dataset (0% churn) should fail
    single_class_df = pd.DataFrame({"Churn": ["No"] * 100})
    res_single = validator._validate_class_imbalance(single_class_df)
    assert res_single["passed"] is False
