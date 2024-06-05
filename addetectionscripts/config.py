# config.py
import logging
import os
import sys
from pathlib import Path

import mlflow

import pretty_errors

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
MLFLOW_TRACKING_URI = "file:///" + str(MODEL_REGISTRY.absolute())  # Three '/' as we're on windows
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


training_config = {
    "xgb_params": {
        ## These params will use get_params so that we can use optuna trial suggestions
        "booster": ("categorical", ["gbtree", "dart"]),
        "lambda": ("float", 1e-8, 1.0, "log"),
        "alpha": ("float", 1e-8, 1.0, "log"),
        "max_depth": ("int", 1, 5),
        "eta": ("float", 1e-8, 1.0, "log"),
        "gamma": ("float", 1e-8, 1.0, "log"),
        "grow_policy": ("categorical", ["depthwise", "lossguide"]),
        ## Below were the optuna hparam search spaces
        # 'booster': trial.suggest_categorical("booster", ["gbtree", "dart"]),
        # 'lambda': trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # 'alpha': trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # 'max_depth': trial.suggest_int("max_depth", 1, 5),
        # 'eta': trial.suggest_float("eta", 1e-8, 1.0, log=True),
        # 'gamma': trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        # 'grow_policy': trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    },
    # Just have 1-1 for testing for now.
    "ray_params": {"num_actors": 1, "cpus_per_actor": 1},
    "ASHAScheduler_params": {"max_t": 10, "grace_period": 1, "reduction_factor": 2},
    "early_stopping_rounds": 10,
    "num_boost_round": 50,
}

transformations_config = {
    "add_groupby_user_features": {
        "grouping_categories": [
            ["ip", "channel"],
            ["ip", "device"],
            ["ip", "os"],
            ["ip", "app"],
            ["ip", "day", "hour"],
            ["app", "channel"],
            ["ip", "app", "os"],
            ["ip", "device", "os", "app"],
        ],
        "grouping_functions": ["nunique", "cumcount"],
    },
    "add_next_click": {"max_num_cats": 2**26},
    "log_bin_column": {"collist": ["next_click"]},
}
