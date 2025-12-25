# Movie Recommendation System

Movie recommendation engine using collaborative filtering, matrix factorization, and neural networks.

## Overview

Complete recommendation system with multiple algorithms:
- Collaborative filtering (user-based and item-based)
- Matrix factorization (SVD, NMF)
- Neural collaborative filtering (deep learning)
- Content-based filtering
- Hybrid recommendation strategies
- Real-time API with Redis caching
- Interactive web dashboard
- Docker deployment

## Architecture

```
Data Layer (MovieLens, ratings, metadata)
    │
Model Training (CF, Matrix Factorization, Neural CF, Content-based)
    │
Recommendation Engine (Hybrid algorithm, cold-start handling)
    │
API Layer (FastAPI - recommendations, similar items, preferences)
    │
Caching (Redis - model cache, embeddings)
    │
Dashboard (Streamlit - interactive UI)
```

## Features

- Multiple algorithms (user/item-based CF, SVD, NMF, Neural CF)
- Hybrid system combining collaborative and content-based
- Cold-start handling for new users/items
- Fast inference with Redis caching
- Evaluation metrics (Precision@K, Recall@K, MAP, NDCG)
- A/B testing framework
- Interactive web dashboard
- Docker deployment with database integration

## 📁 Project Structure

```
04_Recommendation_System/
├── README.md                 # This file
├── PROJECT_PROPOSAL.md       # Detailed project proposal
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker container definition
├── docker-compose.yml       # Docker Compose configuration
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py        # Data loading and preprocessing
│   │   └── preprocessor.py  # Data preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── collaborative_filtering.py  # User/Item-based CF
│   │   ├── matrix_factorization.py     # SVD, NMF
│   │   ├── neural_cf.py                # Neural Collaborative Filtering
│   │   ├── content_based.py            # Content-based filtering
│   │   └── hybrid.py                   # Hybrid recommendation
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py      # Precision@K, Recall@K, NDCG, MAP
│   │   └── evaluation.py   # Model evaluation framework
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py         # FastAPI backend
│   ├── config/
│   │   └── settings.py     # Configuration settings
│   └── utils/
│       ├── __init__.py
│       └── cache.py        # Redis caching utilities
├── app/
│   └── streamlit_app.py    # Streamlit dashboard
├── notebooks/
│   └── exploration.ipynb   # Data exploration notebook
├── data/                   # Dataset files
├── models/                 # Trained models
└── results/                # Evaluation results

```

## Quick Start

**Prerequisites:** Python 3.8+, Docker (optional), PostgreSQL, Redis

**Install and Run:**
```bash
cd 04_Recommendation_System
pip install -r requirements.txt

# Download dataset
python src/data/download_dataset.py

# Train models
python src/models/train_all.py

# Start API
python -m uvicorn src.api.main:app --reload

# Start dashboard
streamlit run app/streamlit_app.py
```

**Docker:**
```bash
docker-compose up -d
```

## Algorithms

**1. Collaborative Filtering**
- User-based: Find similar users, recommend their liked items
- Item-based: Find similar items based on user history

**2. Matrix Factorization**
- SVD: Singular Value Decomposition for latent factors
- NMF: Non-negative Matrix Factorization

**3. Neural Collaborative Filtering**
- Deep learning for user-item interactions
- Learned embeddings for users and items

**4. Content-Based**
- Feature engineering from movie metadata
- Cosine similarity on feature vectors

**5. Hybrid**
- Weighted combination of multiple algorithms
- Adaptive selection based on data availability

## Evaluation Metrics

- **Precision@K** - Fraction of recommended items that are relevant
- **Recall@K** - Fraction of relevant items that are recommended
- **MAP** - Mean Average Precision across users
- **NDCG** - Normalized Discounted Cumulative Gain (ranking quality)

## Testing

```bash
pytest tests/
```

## License

MIT