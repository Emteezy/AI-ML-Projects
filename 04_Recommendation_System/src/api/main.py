"""
FastAPI Backend for Recommendation System
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pickle
import os
from pathlib import Path

from src.config import MODELS_DIR, DEFAULT_N_RECOMMENDATIONS, MAX_RECOMMENDATIONS

app = FastAPI(
    title="Movie Recommendation System API",
    description="API for movie recommendations using multiple algorithms",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
models = {}


# Request/Response models
class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID")
    n_recommendations: int = Field(
        default=DEFAULT_N_RECOMMENDATIONS,
        ge=1,
        le=MAX_RECOMMENDATIONS,
        description="Number of recommendations"
    )
    algorithm: Optional[str] = Field(
        default="hybrid",
        description="Algorithm to use: 'user_cf', 'item_cf', 'svd', 'nmf', 'content_based', 'neural_cf', 'hybrid'"
    )


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[dict]
    algorithm: str


class PredictionRequest(BaseModel):
    user_id: int
    item_id: int
    algorithm: Optional[str] = "hybrid"


class PredictionResponse(BaseModel):
    user_id: int
    item_id: int
    predicted_rating: float
    algorithm: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]


@app.on_event("startup")
async def load_models():
    """Load trained models on startup"""
    global models
    
    model_files = {
        'user_cf': 'user_cf_model.pkl',
        'item_cf': 'item_cf_model.pkl',
        'svd': 'svd_model.pkl',
        'nmf': 'nmf_model.pkl',
        'content_based': 'content_based_model.pkl',
        'neural_cf': 'neural_cf_model.pkl',
        'hybrid': 'hybrid_model.pkl'
    }
    
    for model_name, filename in model_files.items():
        model_path = MODELS_DIR / filename
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
                print(f"Loaded {model_name} model")
            except Exception as e:
                print(f"Error loading {model_name} model: {e}")
        else:
            print(f"Model file not found: {model_path}")


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys())
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys())
    }


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """Get recommendations for a user"""
    algorithm = request.algorithm.lower()
    
    if algorithm not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{algorithm}' not found. Available models: {list(models.keys())}"
        )
    
    model = models[algorithm]
    
    try:
        recommendations = model.recommend(
            user_id=request.user_id,
            n_recommendations=request.n_recommendations,
            exclude_rated=True
        )
        
        # Format recommendations
        rec_list = [
            {
                "item_id": item_id,
                "predicted_rating": float(rating)
            }
            for item_id, rating in recommendations
        ]
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=rec_list,
            algorithm=algorithm
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict rating for a user-item pair"""
    algorithm = request.algorithm.lower()
    
    if algorithm not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{algorithm}' not found. Available models: {list(models.keys())}"
        )
    
    model = models[algorithm]
    
    try:
        predicted_rating = model.predict(
            user_id=request.user_id,
            item_id=request.item_id
        )
        
        return PredictionResponse(
            user_id=request.user_id,
            item_id=request.item_id,
            predicted_rating=float(predicted_rating),
            algorithm=algorithm
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")


@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": list(models.keys()),
        "models_dir": str(MODELS_DIR)
    }


if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT, API_RELOAD
    
    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD
    )

