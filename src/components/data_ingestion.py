import os
import sys
from dataclasses import dataclass

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import get_logger
from src import constants as C

logger = get_logger(__name__)


@dataclass
class DataIngestionConfig:
    source_url: str = os.getenv("DATA_SOURCE_URL", C.RAW_DATA_FILE)
    raw_data_path: str = C.RAW_DATA_FILE
    train_data_path: str = C.INGESTED_TRAIN_FILE
    test_data_path: str = C.INGESTED_TEST_FILE
    test_size: float = C.TEST_SIZE
    random_state: int = C.RANDOM_STATE


class DataIngestion:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    def _download_if_needed(self) -> None:
        os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
        if os.path.exists(self.config.raw_data_path):
            logger.info(f"Raw data already present at {self.config.raw_data_path}")
            return

        if self.config.source_url.startswith("http"):
            logger.info(f"Downloading dataset from {self.config.source_url}")
            resp = requests.get(self.config.source_url, timeout=30)
            resp.raise_for_status()
            with open(self.config.raw_data_path, "wb") as f:
                f.write(resp.content)
        else:
            raise FileNotFoundError(
                f"No local file at {self.config.raw_data_path} and DATA_SOURCE_URL is not a URL."
            )

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

            logger.info(
                f"Data ingestion complete. train={train_set.shape} test={test_set.shape}"
            )
            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    DataIngestion().initiate_data_ingestion()
