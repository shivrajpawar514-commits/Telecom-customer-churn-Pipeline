import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_object, save_numpy_array, read_yaml
from src import constants as C

logger = get_logger(__name__)


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path: str = C.PREPROCESSOR_OBJ_PATH
    transformed_train_path: str = C.TRANSFORMED_TRAIN_FILE
    transformed_test_path: str = C.TRANSFORMED_TEST_FILE
    schema_path: str = C.SCHEMA_FILE_PATH


class DataTransformation:
    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self.config = config
        self.schema = read_yaml(self.config.schema_path)

    def get_preprocessor(self) -> ColumnTransformer:
        num_cols = self.schema["numerical_columns"]
        cat_cols = self.schema["categorical_columns"]

        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        return ColumnTransformer([
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ])

    def initiate_data_transformation(self, train_path: str, test_path: str):
        logger.info("Starting data transformation")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            drop_cols = self.schema.get("drop_columns", [])
            train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
            test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

            target_col = self.schema["target_column"]
            train_df[target_col] = train_df[target_col].map({"Yes": 1, "No": 0}).astype(int)
            test_df[target_col] = test_df[target_col].map({"Yes": 1, "No": 0}).astype(int)

            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col].values
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col].values

            preprocessor = self.get_preprocessor()
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            if hasattr(X_train_arr, "toarray"):
                X_train_arr = X_train_arr.toarray()
                X_test_arr = X_test_arr.toarray()

            save_object(self.config.preprocessor_obj_path, preprocessor)
            save_numpy_array(self.config.transformed_train_path, X=X_train_arr, y=y_train)
            save_numpy_array(self.config.transformed_test_path, X=X_test_arr, y=y_test)

            logger.info(
                f"Data transformation complete. train={X_train_arr.shape} test={X_test_arr.shape}"
            )
            return (
                X_train_arr, y_train, X_test_arr, y_test,
                self.config.preprocessor_obj_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    DataTransformation().initiate_data_transformation(C.INGESTED_TRAIN_FILE, C.INGESTED_TEST_FILE)
