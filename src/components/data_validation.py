import sys
from dataclasses import dataclass, field

import pandas as pd

from src.exception import CustomException
from src.logger import get_logger
from src.utils import read_yaml, save_json
from src import constants as C

logger = get_logger(__name__)


@dataclass
class DataValidationConfig:
    schema_path: str = C.SCHEMA_FILE_PATH
    report_path: str = C.VALIDATION_REPORT_PATH


class DataValidation:
    def __init__(self, config: DataValidationConfig = DataValidationConfig()):
        self.config = config
        self.schema = read_yaml(self.config.schema_path)

    def _validate_columns(self, df: pd.DataFrame) -> dict:
        expected_cols = set(self.schema["columns"].keys())
        actual_cols = set(df.columns)
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols
        return {
            "missing_columns": sorted(missing),
            "unexpected_columns": sorted(extra),
            "passed": len(missing) == 0,
        }

    def _validate_dtypes(self, df: pd.DataFrame) -> dict:
        mismatches = {}
        for col, expected_dtype in self.schema["columns"].items():
            if col not in df.columns:
                continue
            actual_dtype = str(df[col].dtype)
            # tolerate int64/float64 coercion, that's fixed in transformation
            if expected_dtype != actual_dtype and not (
                expected_dtype in ("int64", "float64") and actual_dtype in ("int64", "float64")
            ):
                mismatches[col] = {"expected": expected_dtype, "actual": actual_dtype}
        return {"mismatches": mismatches, "passed": len(mismatches) == 0}

    def _validate_missing_values(self, df: pd.DataFrame) -> dict:
        ratios = (df.isnull().sum() / len(df)).to_dict()
        threshold = self.schema.get("max_missing_ratio", 0.3)
        offending = {c: r for c, r in ratios.items() if r > threshold}
        return {"missing_ratios": ratios, "offending_columns": offending, "passed": len(offending) == 0}

    def initiate_data_validation(self, train_path: str, test_path: str) -> dict:
        logger.info("Starting data validation")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            report = {
                "train": {
                    "columns": self._validate_columns(train_df),
                    "dtypes": self._validate_dtypes(train_df),
                    "missing_values": self._validate_missing_values(train_df),
                },
                "test": {
                    "columns": self._validate_columns(test_df),
                    "dtypes": self._validate_dtypes(test_df),
                    "missing_values": self._validate_missing_values(test_df),
                },
            }

            overall_passed = all(
                report[split][check]["passed"]
                for split in ("train", "test")
                for check in ("columns", "missing_values")
            )
            report["validation_status"] = overall_passed

            save_json(self.config.report_path, report)
            logger.info(f"Data validation status: {overall_passed}")

            if not overall_passed:
                raise ValueError(f"Data validation failed. See {self.config.report_path}")

            return report

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    DataValidation().initiate_data_validation(C.INGESTED_TRAIN_FILE, C.INGESTED_TEST_FILE)
