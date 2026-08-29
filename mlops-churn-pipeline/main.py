from src.logger import logger
from src.exception import CustomException
import sys

logger.info("Project Started")

try:
    a = 10
    b = 0

    print(a / b)

except Exception as e:
    raise CustomException(e, sys)