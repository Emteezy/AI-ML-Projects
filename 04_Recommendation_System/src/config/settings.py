"""
Configuration settings for the recommendation system
"""
import os
from pathlib import Path
from typing import Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Dataset configuration
DATASET_NAME = "movielens"
DATASET_SIZE = "100k"  # Options: "100k", "1m", "10m", "20m"
DATASET_URL = {
    "100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "10m": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
}

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_RATINGS_PER_USER = 5
MIN_RATINGS_PER_ITEM = 5

# Collaborative Filtering
CF_SIMILARITY_METRIC = "cosine"  # Options: "cosine", "pearson", "euclidean"
CF_NEIGHBORS = 50

# Matrix Factorization
MF_FACTORS = 50
MF_EPOCHS = 20
MF_REGULARIZATION = 0.01
MF_LEARNING_RATE = 0.01

# Neural Collaborative Filtering
NCF_EMBEDDING_DIM = 50
NCF_LAYERS = [64, 32, 16]
NCF_EPOCHS = 10
NCF_BATCH_SIZE = 256
NCF_LEARNING_RATE = 0.001

# Content-Based
CB_FEATURE_WEIGHTS = {
    "genres": 1.0,
    "year": 0.5,
    "director": 0.3,
}

# Hybrid System
HYBRID_WEIGHTS = {
    "collaborative_filtering": 0.4,
    "matrix_factorization": 0.3,
    "content_based": 0.2,
    "neural_cf": 0.1,
}

# Evaluation
EVAL_K_VALUES = [5, 10, 20]
EVAL_METRICS = ["precision", "recall", "ndcg", "map"]

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"

# Redis Configuration (optional)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
USE_REDIS = os.getenv("USE_REDIS", "false").lower() == "true"

# Streamlit Configuration
STREAMLIT_PORT = 8501

# Recommendation
DEFAULT_N_RECOMMENDATIONS = 10
MAX_RECOMMENDATIONS = 100

