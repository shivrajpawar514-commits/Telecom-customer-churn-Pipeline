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

    def _validate_class_imbalance(self, df: pd.DataFrame) -> dict:
        target_col = self.schema.get("target_column", C.TARGET_COLUMN)
        if target_col not in df.columns:
            return {"passed": True, "details": "Target column not present"}
        counts = df[target_col].value_counts().to_dict()
        total = len(df)
        ratios = {str(k): round(v / total, 4) for k, v in counts.items()}
        # Verify both classes exist and positive ratio is between 5% and 95%
        pos_key = "Yes" if "Yes" in counts else (1 if 1 in counts else None)
        pos_ratio = ratios.get(str(pos_key), 0.0) if pos_key is not None else 0.0
        balanced = len(counts) >= 2 and 0.05 <= pos_ratio <= 0.95
        return {
            "class_counts": counts,
            "class_ratios": ratios,
            "positive_class_ratio": pos_ratio,
            "passed": balanced,
        }

    def _validate_numerical_ranges(self, df: pd.DataFrame) -> dict:
        violations = {}
        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            if col in df.columns:
                negatives = (pd.to_numeric(df[col], errors="coerce") < 0).sum()
                if negatives > 0:
                    violations[col] = int(negatives)
        return {"negative_value_counts": violations, "passed": len(violations) == 0}

    def validate_dataframe(self, df: pd.DataFrame) -> dict:
        """Validate a single dataframe against schema, missing values, dtypes, and ranges."""
        cols = self._validate_columns(df)
        dtypes = self._validate_dtypes(df)
        missing = self._validate_missing_values(df)
        imbalance = self._validate_class_imbalance(df)
        ranges = self._validate_numerical_ranges(df)
        passed = cols["passed"] and missing["passed"] and dtypes["passed"] and ranges["passed"]
        return {
            "columns": cols,
            "dtypes": dtypes,
            "missing_values": missing,
            "class_imbalance": imbalance,
            "numerical_ranges": ranges,
            "passed": passed,
        }

    def initiate_data_validation(self, train_path: str, test_path: str) -> dict:
        logger.info("Starting data validation")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_val = self.validate_dataframe(train_df)
            test_val = self.validate_dataframe(test_df)

            report = {
                "train": train_val,
                "test": test_val,
            }

            overall_passed = train_val["passed"] and test_val["passed"]
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
