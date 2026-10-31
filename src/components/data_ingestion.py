import os
import sys
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import get_logger
from src.utils import compute_file_hash, save_json
from src import constants as C

load_dotenv()
logger = get_logger(__name__)


def generate_synthetic_telco_churn(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Deterministic synthetic data generator for offline/air-gapped reproducibility."""
    rng = np.random.RandomState(random_state)
    contracts = ["Month-to-month", "One year", "Two year"]
    internets = ["DSL", "Fiber optic", "No"]
    payments = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    yes_no = ["No", "Yes"]

    genders = rng.choice(["Male", "Female"], size=n_samples)
    senior = rng.choice([0, 1], p=[0.84, 0.16], size=n_samples)
    partner = rng.choice(yes_no, size=n_samples)
    dependents = rng.choice(yes_no, size=n_samples)
    tenure = rng.randint(1, 73, size=n_samples)
    phone = rng.choice(["Yes", "No"], p=[0.9, 0.1], size=n_samples)
    multiple = rng.choice(["No", "Yes", "No phone service"], size=n_samples)
    internet = rng.choice(internets, size=n_samples)
    online_sec = rng.choice(yes_no, size=n_samples)
    online_bkp = rng.choice(yes_no, size=n_samples)
    device_prot = rng.choice(yes_no, size=n_samples)
    tech_supp = rng.choice(yes_no, size=n_samples)
    stream_tv = rng.choice(yes_no, size=n_samples)
    stream_mov = rng.choice(yes_no, size=n_samples)
    contract = rng.choice(contracts, p=[0.55, 0.25, 0.20], size=n_samples)
    paperless = rng.choice(yes_no, size=n_samples)
    payment = rng.choice(payments, size=n_samples)
    monthly = np.round(rng.uniform(18.25, 118.75, size=n_samples), 2)
    total = np.round(monthly * tenure + rng.normal(0, 10, size=n_samples), 2)
    total = np.maximum(total, monthly)

    # Churn probability higher for month-to-month and higher charges
    prob = 0.15 + (contract == "Month-to-month") * 0.25 + (monthly > 70) * 0.15 - (tenure > 36) * 0.20
    prob = np.clip(prob, 0.05, 0.90)
    churn = ["Yes" if p > 0.45 else "No" for p in prob]

    cust_ids = [f"{rng.randint(1000, 9999)}-SYNTH" for _ in range(n_samples)]

    df = pd.DataFrame({
        "customerID": cust_ids,
        "gender": genders,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_bkp,
        "DeviceProtection": device_prot,
        "TechSupport": tech_supp,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_mov,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": churn,
    })
    return df


@dataclass
class DataIngestionConfig:
    source_url: str = os.getenv("DATA_SOURCE_URL", C.DEFAULT_DATA_SOURCE_URL)
    raw_data_path: str = C.RAW_DATA_FILE
    train_data_path: str = C.INGESTED_TRAIN_FILE
    test_data_path: str = C.INGESTED_TEST_FILE
    dataset_version_path: str = C.DATASET_VERSION_PATH
    test_size: float = C.TEST_SIZE
    random_state: int = C.RANDOM_STATE


class DataIngestion:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    def _download_if_needed(self) -> None:
        os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
        if os.path.exists(self.config.raw_data_path) and os.path.getsize(self.config.raw_data_path) > 0:
            logger.info(f"Raw data already present at {self.config.raw_data_path}")
            return

        if self.config.source_url.startswith("http"):
            try:
                logger.info(f"Downloading dataset from {self.config.source_url}")
                resp = requests.get(self.config.source_url, timeout=30)
                resp.raise_for_status()
                with open(self.config.raw_data_path, "wb") as f:
                    f.write(resp.content)
                logger.info("Dataset downloaded successfully.")
                return
            except Exception as e:
                logger.warning(
                    f"Failed to download from {self.config.source_url}: {e}. "
                    "Falling back to deterministic synthetic dataset generator for clean reproducibility."
                )
        else:
            logger.warning(
                f"Source {self.config.source_url} is not an HTTP URL. "
                "Generating deterministic reproducible dataset."
            )

        synth_df = generate_synthetic_telco_churn(n_samples=1000, random_state=self.config.random_state)
        synth_df.to_csv(self.config.raw_data_path, index=False)
        logger.info(f"Saved synthetic reproducible dataset to {self.config.raw_data_path}")

    def initiate_data_ingestion(self) -> tuple[str, str]:
        logger.info("Starting data ingestion")
        try:
            self._download_if_needed()

            df = pd.read_csv(self.config.raw_data_path)
            logger.info(f"Loaded raw dataset with shape {df.shape}")

            # TotalCharges sometimes ships as blank-string for new customers
            if "TotalCharges" in df.columns:
                df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            stratify_col = df[C.TARGET_COLUMN] if C.TARGET_COLUMN in df.columns else None
            train_set, test_set = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify_col,
            )

            train_set.to_csv(self.config.train_data_path, index=False)
            test_set.to_csv(self.config.test_data_path, index=False)

            # Record dataset versioning metadata
            raw_hash = compute_file_hash(self.config.raw_data_path)
            train_hash = compute_file_hash(self.config.train_data_path)
            test_hash = compute_file_hash(self.config.test_data_path)

            version_meta = {
                "dataset_name": "IBM-Telco-Customer-Churn",
                "raw_data_sha256": raw_hash,
                "train_data_sha256": train_hash,
                "test_data_sha256": test_hash,
                "source": self.config.source_url,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "random_state": self.config.random_state,
                "test_size": self.config.test_size,
                "total_records": len(df),
                "train_records": len(train_set),
                "test_records": len(test_set),
                "feature_columns": [c for c in df.columns if c != C.TARGET_COLUMN],
                "target_column": C.TARGET_COLUMN,
                "class_distribution": {
                    "total": df[C.TARGET_COLUMN].value_counts().to_dict() if C.TARGET_COLUMN in df.columns else {},
                    "train": train_set[C.TARGET_COLUMN].value_counts().to_dict() if C.TARGET_COLUMN in train_set.columns else {},
                    "test": test_set[C.TARGET_COLUMN].value_counts().to_dict() if C.TARGET_COLUMN in test_set.columns else {},
                },
            }
            save_json(self.config.dataset_version_path, version_meta)
            logger.info(f"Dataset version manifest saved to {self.config.dataset_version_path}")

            logger.info(
                f"Data ingestion complete. train={train_set.shape} test={test_set.shape}"
            )
            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    DataIngestion().initiate_data_ingestion()
