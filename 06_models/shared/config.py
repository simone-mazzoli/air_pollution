import numpy as np
import torch

from . import paths

# Current experiment
MODEL = "resnet"
POLLUTANTS = ["pm25"]
S5P_STREAMS = ["no2", "co"]
USE_AER_WIDE = True
USE_DEM = True

# Data and leakage control
BATCH_SIZE = 128
BUFFER_KM = 100.0
MAX_PM10 = 120.0
MAX_PM25 = 80.0

# Training and evaluation
SEED = 123
CUDNN_DETERMINISTIC = False
CV_EPOCHS = 500
CV_PATIENCE = 25
FINAL_EPOCHS = 25
USE_TTA = True

# Script-specific defaults
CV_FOLDS = None


def result_paths(experiment):
    result_dir = paths.RESULTS / experiment
    return {
        "dir": result_dir,
        "cv_results": result_dir / "eea_cv_results.json",
        "cv_predictions": result_dir / "eea_cv_predictions.csv",
        "final_checkpoint": result_dir / "final_model.pt",
        "test_predictions": result_dir / "test_predictions.csv",
    }


RESULT_PATHS = result_paths("resnet_frozen")
CV_RESULTS = RESULT_PATHS["cv_results"]
CV_PREDICTIONS = RESULT_PATHS["cv_predictions"]
FINAL_CHECKPOINT = RESULT_PATHS["final_checkpoint"]
TEST_PREDICTIONS = RESULT_PATHS["test_predictions"]

# Fixed normalization constants
RELIEF_SCALE = 250.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISPLAY = {"pm10": "PM10", "pm25": "PM2.5"}

MEAN = np.array(
    [438.3721, 614.0557, 588.4096, 942.8433, 1769.9316,
     2049.5515, 2193.292, 2235.5566, 1568.2268, 997.7325],
    dtype="float32",
).reshape(1, 1, -1)
STD = np.array(
    [607.0269, 603.2968, 684.5688, 738.4327, 1100.4561,
     1275.8054, 1369.3717, 1356.5441, 1070.1613, 813.5276],
    dtype="float32",
).reshape(1, 1, -1)

SHARED_EXPERIMENT_CONFIG = {
    "batch": BATCH_SIZE,
    "s5p_streams": S5P_STREAMS,
    "use_aer_wide": USE_AER_WIDE,
    "use_dem": USE_DEM,
    "tta": USE_TTA,
    "buffer_km": BUFFER_KM,
    "max_pm10": MAX_PM10,
    "max_pm25": MAX_PM25,
    "pollutants": POLLUTANTS,
}


def training_config(model_config, *, epochs, folds=CV_FOLDS):
    cfg = dict(SHARED_EXPERIMENT_CONFIG)
    cfg.update(model_config)
    cfg["epochs"] = epochs
    cfg["patience"] = CV_PATIENCE
    cfg["folds"] = folds
    cfg["pollutants"] = list(POLLUTANTS)
    cfg["s5p_streams"] = list(S5P_STREAMS)
    return cfg


# Backwards-compatible name for older imports and notebooks.
CONFIG = training_config({}, epochs=CV_EPOCHS)
