# config.py
import logging
import os
import sys
from pathlib import Path

import mlflow

# Directories
ROOT_DIR = Path(__file__).parent.parent.absolute()
LOGS_DIR = Path(ROOT_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
## EFS DIR for production, comment out for testing now
# EFS_DIR = Path(f"/efs/shared_storage/addetectionscripts/{os.environ.get('GITHUB_USERNAME', '')}")
# try:
#     Path(EFS_DIR).mkdir(parents=True, exist_ok=True)
# except OSError:
#     EFS_DIR = Path(ROOT_DIR, "efs")
#     Path(EFS_DIR).mkdir(parents=True, exist_ok=True)

# Config MLflow
# MODEL_REGISTRY = Path(f"{EFS_DIR}/mlflow") # EFS for prod, test locally first
MODEL_REGISTRY = Path("/tmp/mlflow")
Path(MODEL_REGISTRY).mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_URI = "file:///" + str(MODEL_REGISTRY.absolute()) # Three '/' as we're on windows
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Logger
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {"format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger()