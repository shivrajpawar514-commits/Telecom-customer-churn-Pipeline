"""
Data + prediction drift monitoring for the deployed churn model.

Compares a reference dataset (e.g. training data) against current production
traffic (e.g. logged requests) using Evidently, and writes an HTML report.

Intended to run on a schedule (cron / AWS EventBridge -> Lambda / ECS task)
against a rolling window of logged inference requests.
"""
import sys
import os
from datetime import datetime

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

from src.exception import CustomException
from src.logger import get_logger
from src import constants as C

logger = get_logger(__name__)

REPORTS_DIR = os.path.join("artifacts", "monitoring")


def run_drift_check(reference_path: str, current_path: str) -> str:
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)

        reference = pd.read_csv(reference_path)
        current = pd.read_csv(current_path)

        report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
        report.run(reference_data=reference, current_data=current)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(REPORTS_DIR, f"drift_report_{timestamp}.html")
        report.save_html(out_path)

        result = report.as_dict()
        drift_detected = result["metrics"][0]["result"].get("dataset_drift", False)

        logger.info(f"Drift check complete. drift_detected={drift_detected} report={out_path}")
        if drift_detected:
            logger.warning(
                "Data drift detected — consider retraining the model "
                "and/or alerting via CloudWatch/SNS."
            )
        return out_path

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_drift_check(C.INGESTED_TRAIN_FILE, C.INGESTED_TEST_FILE)
