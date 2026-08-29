import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s ] %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("MLOpsLogger")

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

