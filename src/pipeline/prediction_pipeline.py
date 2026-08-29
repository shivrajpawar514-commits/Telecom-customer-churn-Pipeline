import sys
import pandas as pd

from src.exception import CustomException
from src.logger import get_logger
from src.utils import load_object
from src import constants as C

logger = get_logger(__name__)


class CustomData:
    """Wraps a single raw customer record into a DataFrame matching the
    training schema (pre one-hot / pre-scaling)."""

    def __init__(self, **kwargs):
        self.data = kwargs

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.data])


class PredictPipeline:
    def __init__(
        self,
        model_path: str = C.MODEL_FILE_PATH,
        preprocessor_path: str = C.PREPROCESSOR_OBJ_PATH,
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self._model = None
        self._preprocessor = None

    def _load(self):
        if self._model is None:
            self._model = load_object(self.model_path)
        if self._preprocessor is None:
            self._preprocessor = load_object(self.preprocessor_path)

    def predict(self, features: pd.DataFrame) -> dict:
        try:
            self._load()
            transformed = self._preprocessor.transform(features)
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()

            pred = self._model.predict(transformed)[0]
            proba = self._model.predict_proba(transformed)[0][1]

            return {
                "churn_prediction": "Yes" if int(pred) == 1 else "No",
                "churn_probability": round(float(proba), 4),
            }
        except Exception as e:
            raise CustomException(e, sys)
