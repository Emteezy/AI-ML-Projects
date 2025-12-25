"""
Configuration settings for the medical image classification API.
"""
import os
from pathlib import Path
from typing import Optional

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_NAME: str = os.getenv("MODEL_NAME", "resnet18")
MODEL_VERSION: str = os.getenv("MODEL_VERSION", "v1")
MODEL_PATH: Optional[str] = os.getenv(
    "MODEL_PATH", str(MODELS_DIR / "best_model.pth")
)

# Image configuration
IMAGE_SIZE: int = 224
IMAGE_MEAN: tuple = (0.485, 0.456, 0.406)  # ImageNet normalization
IMAGE_STD: tuple = (0.229, 0.224, 0.225)

# Class labels
CLASS_LABELS: list = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES: int = 2

# API configuration
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"

# Dataset paths
DATASET_PATH: Path = DATA_DIR / "chest_xray"
TRAIN_DIR: Path = DATASET_PATH / "train"
VAL_DIR: Path = DATASET_PATH / "val"
TEST_DIR: Path = DATASET_PATH / "test"

# Training configuration
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.001"))
NUM_EPOCHS: int = int(os.getenv("NUM_EPOCHS", "10"))
DEVICE: str = os.getenv("DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")

# Logging
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: Optional[str] = os.getenv("LOG_FILE", str(BASE_DIR / "logs" / "app.log"))

