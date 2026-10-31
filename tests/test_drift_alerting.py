import os
import json
import pytest
import pandas as pd

from monitoring.monitor_drift import run_drift_check, evaluate_drift_severity
from src import constants as C


def test_evaluate_drift_severity_tiers():
    """Verify drift severity is categorized correctly into tiers."""
    assert evaluate_drift_severity(drift_share=0.0, drifted_features=[], target_drift=False) == "NO_DRIFT"
    assert evaluate_drift_severity(drift_share=0.05, drifted_features=["gender"], target_drift=False) == "INFO"
    assert evaluate_drift_severity(drift_share=0.15, drifted_features=["OnlineBackup"], target_drift=False) == "WARNING"
    assert evaluate_drift_severity(drift_share=0.08, drifted_features=["Contract"], target_drift=False) == "WARNING"
    assert evaluate_drift_severity(drift_share=0.35, drifted_features=["Contract", "tenure"], target_drift=False) == "CRITICAL"
    assert evaluate_drift_severity(drift_share=0.05, drifted_features=[], target_drift=True) == "CRITICAL"


def test_drift_alert_manifest_generation():
    """Verify drift check produces HTML report, JSON alert, and adheres to safety policy."""
    assert os.path.exists(C.INGESTED_TRAIN_FILE), "train.csv must exist"
    assert os.path.exists(C.INGESTED_TEST_FILE), "test.csv must exist"

    report_path = run_drift_check(C.INGESTED_TRAIN_FILE, C.INGESTED_TEST_FILE)
    assert os.path.exists(report_path)
    assert report_path.endswith(".html")

    assert os.path.exists(C.DRIFT_ALERT_PATH), "drift_alert.json must be written"
    with open(C.DRIFT_ALERT_PATH, "r") as f:
        alert = json.load(f)

    assert "alert_id" in alert
    assert "severity" in alert
    assert "drift_share" in alert
    assert "policy" in alert

    # CRITICAL GOVERNANCE SAFETY RULES:
    # Drift detection must NEVER automatically trigger retrain or deploy!
    assert alert["policy"]["automated_retrain"] is False
    assert alert["policy"]["automated_deploy"] is False
    assert "review" in alert["policy"]["action_required"].lower()
