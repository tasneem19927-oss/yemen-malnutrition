"""
Centralized logging configuration.
"""

import logging
import logging.handlers
import os
from datetime import datetime


def setup_logging():
    """Configure application logging."""
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(),
            logging.handlers.RotatingFileHandler(
                f"{log_dir}/app.log",
                maxBytes=10*1024*1024,  # 10MB
                backupCount=10,
            ),
            logging.handlers.TimedRotatingFileHandler(
                f"{log_dir}/app_daily.log",
                when="midnight",
                backupCount=30,
            ),
        ]
    )

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
