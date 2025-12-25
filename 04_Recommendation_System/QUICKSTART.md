# Quick Start Guide

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python src/data/download_dataset.py
```

This will download the MovieLens 100k dataset by default. For larger datasets:

```bash
python src/data/download_dataset.py --size 1m
```

### 3. Train Models

Train all models:

```bash
python src/models/train_all.py
```

Train specific models:

```bash
python src/models/train_all.py --models user_cf item_cf svd
```

Skip evaluation (faster training):

```bash
python src/models/train_all.py --no-eval
```

### 4. Start API Server

```bash
python -m uvicorn src.api.main:app --reload
```

API will be available at: http://localhost:8000

API Documentation: http://localhost:8000/docs

### 5. Start Streamlit Dashboard

In a new terminal:

```bash
streamlit run app/streamlit_app.py
```

Dashboard will be available at: http://localhost:8501

### 6. Using Docker (Optional)

```bash
docker-compose up -d
```

This will start both API and Streamlit services.

## 📝 Example Usage

### API Endpoints

#### Get Recommendations

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "n_recommendations": 10,
    "algorithm": "hybrid"
  }'
```

#### Predict Rating

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "item_id": 50,
    "algorithm": "hybrid"
  }'
```

### Python Usage

```python
import pickle
from src.models.collaborative_filtering import UserBasedCF
from src.data.loader import MovieLensLoader
from src.data.preprocessor import DataPreprocessor

# Load data
loader = MovieLensLoader()
ratings, movies, users = loader.load_all()

# Preprocess
preprocessor = DataPreprocessor()
ratings_filtered = preprocessor.filter_sparse_data(ratings)
ratings_encoded, mappings = preprocessor.encode_ids(ratings_filtered)
user_item_matrix = preprocessor.create_user_item_matrix(ratings_encoded)

# Train model
model = UserBasedCF()
model.fit(user_item_matrix)

# Get recommendations
recommendations = model.recommend(user_id=1, n_recommendations=10)
print(recommendations)
```

## 🐛 Troubleshooting

### Dataset Download Issues

If download fails, try:
```bash
python src/data/download_dataset.py --force
```

### Model Training Errors

- Ensure dataset is downloaded first
- Check that you have enough memory (Neural CF requires more)
- Try training models individually

### API Connection Issues

- Check that API is running: `curl http://localhost:8000/health`
- Verify port 8000 is not in use
- Check firewall settings

### Streamlit Issues

- Ensure API is running first
- Check API_URL in environment variables
- Verify port 8501 is available

## 📊 Next Steps

1. Explore the notebooks in `notebooks/` for data analysis
2. Experiment with different algorithms
3. Evaluate models using the evaluation framework
4. Deploy to production using Docker

