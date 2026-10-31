"""
Data + prediction drift monitoring for the deployed churn model.

Compares a reference dataset (e.g. training data) against current production
traffic (e.g. logged requests) using Evidently, and writes:
1. An interactive HTML drift report (artifacts/monitoring/drift_report_*.html)
2. A machine-readable drift alert manifest (artifacts/monitoring/drift_alert.json)

POLICY & SAFETY GUARDRAIL:
Do NOT automatically retrain or deploy a model just because drift was detected!
Drift may be caused by telemetry bugs, upstream schema corruptions, seasonal
marketing campaigns, or temporary external events. Automated retraining without
human review and label verification can train on poisoned distributions, causing
silent production degradation. Always follow the documented Review SOP
(docs/DRIFT_RESPONSE_AND_ALERTING.md).
"""
import sys
import os
import json
from datetime import datetime

# Ensure project root is in sys.path when executed directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_json
from src import constants as C

logger = get_logger(__name__)

REPORTS_DIR = C.MONITORING_DIR
KEY_CHURN_DRIVERS = ["Contract", "tenure", "MonthlyCharges", "InternetService", "TechSupport"]


def evaluate_drift_severity(drift_share: float, drifted_features: list, target_drift: bool) -> str:
    """Classify drift into actionable severity tiers."""
    key_drivers_drifted = [f for f in drifted_features if f in KEY_CHURN_DRIVERS]
    if drift_share >= 0.30 or target_drift or len(key_drivers_drifted) >= 2:
        return "CRITICAL"
    elif drift_share >= 0.10 or len(key_drivers_drifted) >= 1:
        return "WARNING"
    elif drift_share > 0:
        return "INFO"
    return "NO_DRIFT"


def run_drift_check(reference_path: str, current_path: str, notify: bool = False) -> str:
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)

        reference = pd.read_csv(reference_path)
        current = pd.read_csv(current_path)

        report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
        report.run(reference_data=reference, current_data=current)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_html_path = os.path.join(REPORTS_DIR, f"drift_report_{timestamp}.html")
        report.save_html(out_html_path)

        result = report.as_dict()

        # Extract drift metrics
        data_drift_metric = result["metrics"][0]["result"]
        dataset_drift = data_drift_metric.get("dataset_drift", False)
        drift_share = float(data_drift_metric.get("drift_share", 0.0))
        n_drifted = int(data_drift_metric.get("number_of_drifted_columns", 0))

        # Check target drift if present in metric results
        target_drift = False
        if len(result["metrics"]) > 1:
            target_metric = result["metrics"][1]["result"]
            target_drift = target_metric.get("target_drift", False)

        # Extract per-column drift status
        drift_by_column = data_drift_metric.get("drift_by_columns", {})
        drifted_features = [
            col for col, details in drift_by_column.items()
            if details.get("drift_detected", False)
        ]

        severity = evaluate_drift_severity(drift_share, drifted_features, target_drift)

        alert_payload = {
            "alert_id": f"DRIFT-{timestamp}",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "severity": severity,
            "drift_detected": dataset_drift,
            "target_drift_detected": target_drift,
            "drift_share": round(drift_share, 4),
            "number_of_drifted_features": n_drifted,
            "drifted_features": drifted_features,
            "key_drivers_affected": [f for f in drifted_features if f in KEY_CHURN_DRIVERS],
            "html_report_path": out_html_path,
            "reference_dataset": reference_path,
            "current_dataset": current_path,
            "policy": {
                "automated_retrain": False,
                "automated_deploy": False,
                "action_required": "Follow human-in-the-loop review procedure in docs/DRIFT_RESPONSE_AND_ALERTING.md",
            },
            "recommended_next_steps": [
                "1. Triage: Inspect drift_report HTML to identify shifted distributions and p-values.",
                "2. Verify Data Quality: Rule out pipeline ingestion schema mutations, upstream API glitches, and sensor errors.",
                "3. Label Delay Audit: Confirm whether real ground truth churn labels (30-60 day maturity) are available.",
                "4. Offline Experimentation: Train candidate model in isolated branch, benchmark against baseline champion.",
                "5. Governance Sign-off: Present metrics to Model Governance Lead before promoting to staging/prod.",
            ],
        }

        save_json(C.DRIFT_ALERT_PATH, alert_payload)

        logger.info(
            f"Drift check complete. severity={severity} drift_detected={dataset_drift} "
            f"drift_share={drift_share:.2%} drifted_features={drifted_features}"
        )

        if severity in ("WARNING", "CRITICAL"):
            logger.warning(
                f"[{severity} ALERT] Distribution drift detected in {n_drifted} features! "
                "ALERT RECORDED: Automated retraining/deploy is blocked per safety policy. "
                f"Review report: {out_html_path} and manifest: {C.DRIFT_ALERT_PATH}"
            )
        else:
            logger.info("Distribution stability verified within acceptable thresholds.")

        return out_html_path

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_drift_check(C.INGESTED_TRAIN_FILE, C.INGESTED_TEST_FILE)
