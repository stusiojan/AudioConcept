from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

"""
Configuration file for project.
"""


# Load environment variables from .env file if it exists
load_dotenv()

# Pick a model to train: "VGGish" or "CNN" or "SVM"
MODEL_TO_TRAIN = "SVM"

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATA_PATH = RAW_DATA_DIR / "gtzan"
SAMPLE_AUDIO_DIR = RAW_DATA_DIR / "sample_audio"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Dataset parameters
GTZAN_GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

# NN parameters
LEARNING_RATE = 1e-2  # VGGish 3e-5  # cnn 0.007  # 3e-4
NUM_EPOCHS = 60
MODEL_PATIENCE = 15
WEIGHT_DECAY = 8e-4  # 5e-4
LABEL_SMOOTHING = 0.15
NOISE_LEVEL = 0.002
VGG16_ARCHITECTURE = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    "M",
    # Then flatten
    # Then 4096x4096x1000 linear layers - here 4096x4096x10
]

# SVM parameters
SVM_TEST_SIZE = 0.2  # Proportion of dataset to include in the test split
SVM_RANDOM_STATE = 42
SVM_PARAM_GRID = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto", 0.1, 1],
}

# input audio validation parameters
VALIDATION_PARAMS = {
    "max_duration": 60.0,  # seconds
    "target_sample_rate": 22050,  # Hz
    "required_format": ".wav",
    "required_channels": 1,  # mono
}


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
