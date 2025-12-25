# Movie Recommendation System 🎬🤖

A **production-ready recommendation system** for movies using collaborative filtering, matrix factorization, and deep learning. This project demonstrates recommendation algorithms, real-time inference, and business application ML.

## 🎯 Overview

This project provides a complete recommendation system featuring:
- **Multiple Algorithms**: Collaborative Filtering, Matrix Factorization, Neural Collaborative Filtering
- **Hybrid Recommendations**: Combining multiple strategies for better results
- **Real-time API**: Fast recommendation serving with caching
- **Interactive Dashboard**: Web interface for exploring recommendations
- **Production Deployment**: Docker, PostgreSQL, Redis caching

## 🏗️ Architecture

```
┌─────────────────┐
│  Data Layer     │
│  - MovieLens    │
│  - User Ratings │
│  - Movie Metadata│
└───────┬─────────┘
        │
┌───────▼──────────────────┐
│  Model Training Layer    │
│  - Collaborative Filtering│
│  - Matrix Factorization  │
│  - Neural CF (Deep)      │
│  - Content-Based         │
└───────┬──────────────────┘
        │
┌───────▼──────────────────┐
│  Recommendation Engine   │
│  - Hybrid Algorithm      │
│  - Cold-start Handling   │
│  - Real-time Inference   │
└───────┬──────────────────┘
        │
┌───────▼──────────────────┐
│  API Layer (FastAPI)     │
│  - Recommendation Endpoint│
│  - Similar Items         │
│  - User Preferences      │
└───────┬──────────────────┘
        │
┌───────▼──────────────────┐
│  Caching Layer (Redis)   │
│  - Model Cache           │
│  - User Embeddings       │
└───────┬──────────────────┘
        │
┌───────▼──────────────────┐
│  Dashboard (Streamlit)   │
│  - Interactive UI        │
│  - Recommendation Display│
│  - Similar Movies        │
└──────────────────────────┘
```

## ✨ Key Features

- ✅ **Multiple Recommendation Algorithms**: User-based, Item-based, Matrix Factorization, Neural CF
- ✅ **Hybrid System**: Combining collaborative filtering and content-based approaches
- ✅ **Cold-Start Handling**: Strategies for new users and items
- ✅ **Real-time Inference**: Fast API with Redis caching
- ✅ **Evaluation Metrics**: Precision@K, Recall@K, MAP, NDCG
- ✅ **A/B Testing Framework**: Model comparison infrastructure
- ✅ **Interactive Dashboard**: Web interface for exploring recommendations
- ✅ **Production Ready**: Docker deployment, database integration

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

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (optional)
- PostgreSQL (for production)
- Redis (for caching)

### Installation

1. **Clone and setup**

```bash
cd 04_Recommendation_System
pip install -r requirements.txt
```

2. **Download Dataset**

```bash
python src/data/download_dataset.py
```

3. **Train Models**

```bash
python src/models/train_all.py
```

4. **Start API**

```bash
python -m uvicorn src.api.main:app --reload
```

5. **Start Dashboard**

```bash
streamlit run app/streamlit_app.py
```

### Docker Deployment

```bash
docker-compose up -d
```

## 📊 Algorithms Implemented

### 1. Collaborative Filtering
- **User-based CF**: Find similar users, recommend items they liked
- **Item-based CF**: Find similar items, recommend based on user history

### 2. Matrix Factorization
- **SVD (Singular Value Decomposition)**: Latent factor model
- **NMF (Non-negative Matrix Factorization)**: Non-negative latent factors

### 3. Neural Collaborative Filtering
- **Deep Learning Model**: Neural network for user-item interactions
- **Embedding Layers**: Learn user and item embeddings

### 4. Content-Based Filtering
- **Feature Engineering**: Use movie metadata (genres, year, etc.)
- **Similarity Computation**: Cosine similarity on feature vectors

### 5. Hybrid System
- **Weighted Combination**: Combine multiple algorithms
- **Adaptive Selection**: Choose algorithm based on data availability

## 🎓 Skills Demonstrated

- ✅ Recommendation System Algorithms
- ✅ Collaborative Filtering
- ✅ Matrix Factorization
- ✅ Deep Learning for Recommendations
- ✅ Evaluation Metrics (Precision@K, Recall@K, NDCG, MAP)
- ✅ Cold-start Problem Solving
- ✅ Real-time Inference
- ✅ Production Deployment

## 📈 Evaluation Metrics

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **MAP (Mean Average Precision)**: Average precision across all users
- **NDCG (Normalized Discounted Cumulative Gain)**: Ranking quality metric

## 🎯 Use Cases

- **E-commerce**: Product recommendations
- **Streaming Services**: Content recommendations
- **Social Media**: Friend/content suggestions
- **News Platforms**: Article recommendations

## ⚠️ Important Notes

- This is a **technical demonstration** project
- For production use, consider scalability, privacy, and business requirements
- Dataset used: MovieLens (for demonstration purposes)

## 🚧 Future Enhancements

- [ ] Real-time learning (online updates)
- [ ] Explainable recommendations
- [ ] Multi-armed bandit for exploration
- [ ] Graph-based recommendations
- [ ] Transformer-based models

## 📄 License

This project is open source and available under the MIT License.

---

**Project Status**: ✅ Implementation Complete  
**Last Updated**: 2024-12-21

