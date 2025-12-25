"""Configuration settings for Wearable Health Monitoring System."""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# MQTT Configuration
MQTT_CONFIG = {
    "broker": os.getenv("MQTT_BROKER", "localhost"),
    "port": int(os.getenv("MQTT_PORT", 1883)),
    "username": os.getenv("MQTT_USERNAME", None),
    "password": os.getenv("MQTT_PASSWORD", None),
    "topic_prefix": "wearable/health",
    "qos": 1,
    "keepalive": 60,
}

# Database Configuration
DATABASE_CONFIG = {
    "url": os.getenv(
        "DATABASE_URL",
        "postgresql://user:password@localhost:5432/wearable_db"
    ),
    "pool_size": 10,
    "max_overflow": 20,
}

# Sensor Configuration
SENSOR_CONFIG = {
    "sampling_rate": {
        "heart_rate": 100,  # Hz
        "spo2": 100,  # Hz
        "accelerometer": 50,  # Hz
        "temperature": 1,  # Hz
    },
    "i2c_addresses": {
        "max30102": 0x57,  # Heart rate & SpO2
        "mpu6050": 0x68,  # Accelerometer/Gyroscope
    },
    "simulation_mode": os.getenv("SIMULATION_MODE", "false").lower() == "true",
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "title": "Wearable Health Monitoring API",
    "version": "1.0.0",
}

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "port": 8501,
    "api_url": os.getenv("API_URL", "http://localhost:8000"),
}

# Health Alert Thresholds
HEALTH_THRESHOLDS = {
    "heart_rate": {
        "min_normal": 60,
        "max_normal": 100,
        "min_warning": 50,
        "max_warning": 120,
    },
    "spo2": {
        "min_normal": 95,
        "min_warning": 90,
        "min_critical": 85,
    },
    "temperature": {
        "min_normal": 36.0,
        "max_normal": 37.5,
        "min_warning": 35.0,
        "max_warning": 38.5,
    },
}

# Device Configuration
DEVICE_CONFIG = {
    "device_id": os.getenv("DEVICE_ID", "device_001"),
    "device_type": os.getenv("DEVICE_TYPE", "raspberry_pi"),
}

