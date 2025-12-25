"""
FastAPI application for medical image classification API.
"""
import io
import base64
from pathlib import Path
from typing import Optional
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
from loguru import logger

from ..models.model_utils import load_model, get_model_info
from ..models.resnet_model import MedicalResNet
from ..utils.preprocessing import preprocess_image
from ..utils.explainability import generate_gradcam, visualize_gradcam, create_explanation_image
from ..config.settings import (
    MODEL_PATH,
    MODEL_NAME,
    CLASS_LABELS,
    DEVICE,
    MODEL_VERSION,
)

# Initialize FastAPI app
app = FastAPI(
    title="Medical Image Classification API",
    description="API for chest X-ray pneumonia detection using deep learning",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model: Optional[MedicalResNet] = None
model_metadata: dict = {}


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    prediction: str
    confidence: float
    class_probabilities: dict
    model_version: str


class ExplanationResponse(BaseModel):
    """Response model for explanation endpoint."""
    prediction: str
    confidence: float
    explanation_image: str
    heatmap_overlay: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    model_info: Optional[dict] = None


@app.on_event("startup")
async def load_model_on_startup():
    """Load model on application startup."""
    global model, model_metadata
    
    try:
        if MODEL_PATH and Path(MODEL_PATH).exists():
            logger.info(f"Loading model from {MODEL_PATH}")
            model, model_metadata = load_model(MODEL_PATH, device=DEVICE)
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}")
            logger.warning("API will continue without model. Train and load model before using /predict endpoint.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.warning("API will continue without model.")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Medical Image Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "explain": "/explain",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = model is not None
    
    response = {
        "status": "healthy",
        "model_loaded": model_loaded,
    }
    
    if model_loaded:
        response["model_version"] = MODEL_VERSION
        response["model_info"] = get_model_info(model)
    
    return response


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict image class (Normal or Pneumonia).
    
    Args:
        file: Image file (JPEG, PNG)
        
    Returns:
        Prediction result with confidence scores
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train and load the model first."
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess
        input_tensor = preprocess_image(image, return_tensor=True)
        input_tensor = input_tensor.to(DEVICE)
        
        # Predict
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        # Get results
        predicted_class_idx = predicted_class.item()
        prediction = CLASS_LABELS[predicted_class_idx]
        confidence_score = confidence.item()
        
        # Get all class probabilities
        probs = probabilities[0].cpu().numpy()
        class_probabilities = {
            CLASS_LABELS[i]: float(probs[i]) for i in range(len(CLASS_LABELS))
        }
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence_score,
            class_probabilities=class_probabilities,
            model_version=MODEL_VERSION,
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/explain", response_model=ExplanationResponse)
async def explain(file: UploadFile = File(...)):
    """
    Get prediction with Grad-CAM explanation.
    
    Args:
        file: Image file (JPEG, PNG)
        
    Returns:
        Prediction with explanation visualization
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train and load the model first."
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess
        input_tensor = preprocess_image(image, return_tensor=True)
        input_tensor = input_tensor.to(DEVICE)
        
        # Predict
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        predicted_class_idx = predicted_class.item()
        prediction = CLASS_LABELS[predicted_class_idx]
        confidence_score = confidence.item()
        
        # Generate Grad-CAM
        heatmap = generate_gradcam(model, input_tensor, target_class=predicted_class_idx, device=DEVICE)
        
        # Get original image as numpy array
        image_np = np.array(image.resize((224, 224)))
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        # Create explanation image
        explanation_img = create_explanation_image(
            image_np,
            heatmap,
            prediction,
            confidence_score,
        )
        
        # Create overlay
        overlay = visualize_gradcam(image_np, heatmap)
        overlay_pil = Image.fromarray(overlay)
        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        overlay_base64 = base64.b64encode(buf.getvalue()).decode()
        
        return ExplanationResponse(
            prediction=prediction,
            confidence=confidence_score,
            explanation_image=explanation_img,
            heatmap_overlay=overlay_base64,
        )
    
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    from ..config.settings import API_HOST, API_PORT, API_RELOAD
    
    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
    )

