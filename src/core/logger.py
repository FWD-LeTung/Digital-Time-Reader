import logging
import sys
from pathlib import Path
from src.core.config import settings

# Define log format: Time | Level | Message
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logger():
    # 1. Create 'logs' folder if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 2. Create the logger object
    logger = logging.getLogger("DigitalClockApp")

    # 3. Set log level based on config (DEBUG = show all, INFO = hide details)
    logger.setLevel(logging.DEBUG if settings.debug_mode else logging.INFO)

    # 4. Check if handlers already exist to avoid duplicate logs
    if logger.hasHandlers():
        return logger

    # 5. Handler 1: Print logs to Terminal (Console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(console_handler)

    # 6. Handler 2: Save logs to 'app.log' file
    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(file_handler)

    return logger

# Create global instance
logger = setup_logger()