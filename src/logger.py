import logging
import os
from datetime import datetime

# Define a safe absolute path to the logs directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "..", "logs")

# Ensure logs directory exists before writing the log file
os.makedirs(LOGS_DIR, exist_ok=True)

# Generate unique log filename
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)

# Create logger
logger = logging.getLogger("mlops_logger")
logger.setLevel(logging.INFO)

# File handler setup
file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a')
formatter = logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Prevent duplicate handlers
if not logger.hasHandlers():
    logger.addHandler(file_handler)
