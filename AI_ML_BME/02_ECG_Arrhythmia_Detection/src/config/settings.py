"""Configuration settings for ECG Arrhythmia Detection System."""

import os
from pathlib import Path
from typing import Dict, List

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MITDB_DIR = DATA_DIR / "mitdb"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
MITDB_DIR.mkdir(parents=True, exist_ok=True)

# Signal processing parameters
SIGNAL_CONFIG = {
    "sampling_rate": 360,  # Hz (MIT-BIH standard)
    "signal_length": 3600,  # 10 seconds at 360 Hz
    "window_size": 3600,  # Window size for analysis
    "overlap": 0,  # Overlap between windows (0-1)
}

# Filtering parameters
FILTER_CONFIG = {
    "bandpass_low": 0.5,  # Hz
    "bandpass_high": 40.0,  # Hz
    "notch_freq": 50.0,  # Hz (50 or 60 for powerline noise)
    "notch_q": 30.0,  # Quality factor
}

# Arrhythmia classes
ARRHYTHMIA_CLASSES = [
    "Normal",
    "Atrial Fibrillation",
    "Premature Ventricular Contraction",
    "Supraventricular Tachycardia",
    "Ventricular Tachycardia",
]

# Map MIT-BIH annotation codes to our classes
ANNOTATION_MAP: Dict[str, str] = {
    "N": "Normal",
    "L": "Normal",
    "R": "Normal",
    "e": "Normal",
    "j": "Normal",
    "A": "Atrial Fibrillation",
    "a": "Atrial Fibrillation",
    "J": "Supraventricular Tachycardia",
    "S": "Supraventricular Tachycardia",
    "V": "Premature Ventricular Contraction",
    "E": "Premature Ventricular Contraction",
    "F": "Ventricular Tachycardia",
}

# Model configuration
MODEL_CONFIG = {
    "lstm": {
        "input_size": 1,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "num_classes": len(ARRHYTHMIA_CLASSES),
    },
    "transformer": {
        "input_size": 1,
        "d_model": 128,
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "num_classes": len(ARRHYTHMIA_CLASSES),
    },
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 20,
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "num_workers": 4,
    "pin_memory": True,
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "title": "ECG Arrhythmia Detection API",
    "version": "1.0.0",
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    "port": 8501,
    "api_url": "http://localhost:8000",
}

# Device configuration
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

