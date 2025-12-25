"""FastAPI application for ECG arrhythmia detection API."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from pathlib import Path

from ..config.settings import (
    MODELS_DIR,
    ARRHYTHMIA_CLASSES,
    DEVICE,
    API_CONFIG,
    SIGNAL_CONFIG,
)
from ..models import load_model, get_model
from ..signal_processing import preprocess_ecg_signal, denoise_signal, extract_features


# Global model cache
_model_cache: Dict[str, Tuple[torch.nn.Module, Dict]] = {}


def load_model_if_needed(model_path: Path, model_type: str = "lstm") -> torch.nn.Module:
    """Load model if not already loaded."""
    cache_key = str(model_path)
    if cache_key not in _model_cache:
        model, metadata = load_model(model_path, device=DEVICE, model_type=model_type)
        _model_cache[cache_key] = (model, metadata)
    return _model_cache[cache_key][0]


# Initialize FastAPI app
app = FastAPI(
    title=API_CONFIG["title"],
    version=API_CONFIG["version"],
    description="ECG Arrhythmia Detection API - Real-time classification of ECG signals"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ECGSignalRequest(BaseModel):
    """Request model for ECG signal prediction."""
    signal: List[float] = Field(..., description="ECG signal as list of floats")
    model_name: Optional[str] = Field(default="lstm_best.pth", description="Model name to use")


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    prediction: str = Field(..., description="Predicted arrhythmia class")
    confidence: float = Field(..., description="Confidence score (0-1)")
    class_probabilities: Dict[str, float] = Field(..., description="Probabilities for each class")
    model_version: str = Field(..., description="Model version used")


class AnalysisResponse(BaseModel):
    """Response model for detailed analysis."""
    prediction: str = Field(..., description="Predicted arrhythmia class")
    confidence: float = Field(..., description="Confidence score (0-1)")
    features: Dict[str, float] = Field(..., description="Extracted ECG features")
    signal_quality: str = Field(..., description="Signal quality assessment")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    available_models: List[str] = Field(default_factory=list)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # Check for available models
    available_models = []
    if MODELS_DIR.exists():
        available_models = [f.name for f in MODELS_DIR.glob("*.pth")]
    
    # Check if default model is loaded
    default_model_path = MODELS_DIR / "lstm_best.pth"
    model_loaded = default_model_path.exists() if default_model_path else False
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        model_version="lstm_v1" if model_loaded else None,
        available_models=available_models
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_ecg(request: ECGSignalRequest):
    """
    Classify ECG signal and return arrhythmia prediction.
    
    Args:
        request: ECG signal data
        
    Returns:
        Prediction with confidence scores
    """
    try:
        # Convert signal to numpy array
        signal = np.array(request.signal, dtype=np.float32)
        
        if len(signal) < SIGNAL_CONFIG["window_size"]:
            raise HTTPException(
                status_code=400,
                detail=f"Signal too short. Minimum length: {SIGNAL_CONFIG['window_size']}"
            )
        
        # Preprocess signal
        processed_signal = preprocess_ecg_signal(signal)
        processed_signal = denoise_signal(processed_signal)
        
        # Ensure correct length (take last window_size samples)
        if len(processed_signal) > SIGNAL_CONFIG["window_size"]:
            processed_signal = processed_signal[-SIGNAL_CONFIG["window_size"]:]
        elif len(processed_signal) < SIGNAL_CONFIG["window_size"]:
            # Pad if needed
            padding = SIGNAL_CONFIG["window_size"] - len(processed_signal)
            processed_signal = np.pad(processed_signal, (padding, 0), mode='constant')
        
        # Load model
        model_path = MODELS_DIR / request.model_name
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {request.model_name}"
            )
        
        model = load_model_if_needed(model_path)
        
        # Prepare input tensor
        # Shape: (window_size, 1) for LSTM/Transformer
        signal_tensor = torch.from_numpy(processed_signal).unsqueeze(-1).unsqueeze(0)
        signal_tensor = signal_tensor.to(DEVICE)
        
        # Predict
        model.eval()
        with torch.no_grad():
            output = model(signal_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class_idx].item()
        
        # Get class probabilities
        class_probs = {
            ARRHYTHMIA_CLASSES[i]: float(probabilities[0, i].item())
            for i in range(len(ARRHYTHMIA_CLASSES))
        }
        
        return PredictionResponse(
            prediction=ARRHYTHMIA_CLASSES[predicted_class_idx],
            confidence=confidence,
            class_probabilities=class_probs,
            model_version=request.model_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_ecg(request: ECGSignalRequest):
    """
    Get detailed ECG analysis with features and quality assessment.
    
    Args:
        request: ECG signal data
        
    Returns:
        Detailed analysis with features and signal quality
    """
    try:
        # Convert signal to numpy array
        signal = np.array(request.signal, dtype=np.float32)
        
        if len(signal) < SIGNAL_CONFIG["window_size"]:
            raise HTTPException(
                status_code=400,
                detail=f"Signal too short. Minimum length: {SIGNAL_CONFIG['window_size']}"
            )
        
        # Preprocess signal
        processed_signal = preprocess_ecg_signal(signal)
        processed_signal = denoise_signal(processed_signal)
        
        # Extract features
        features = extract_features(processed_signal)
        
        # Ensure correct length for prediction
        if len(processed_signal) > SIGNAL_CONFIG["window_size"]:
            signal_for_pred = processed_signal[-SIGNAL_CONFIG["window_size"]:]
        else:
            signal_for_pred = processed_signal
        
        # Load model
        model_path = MODELS_DIR / request.model_name
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {request.model_name}"
            )
        
        model = load_model_if_needed(model_path)
        
        # Prepare input tensor
        signal_tensor = torch.from_numpy(signal_for_pred).unsqueeze(-1).unsqueeze(0)
        signal_tensor = signal_tensor.to(DEVICE)
        
        # Predict
        model.eval()
        with torch.no_grad():
            output = model(signal_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class_idx].item()
        
        # Assess signal quality
        signal_std = np.std(processed_signal)
        if signal_std < 0.1:
            quality = "poor"
        elif signal_std < 0.5:
            quality = "fair"
        else:
            quality = "good"
        
        # Include key features
        key_features = {
            "heart_rate": features.get("heart_rate", 0.0),
            "qrs_duration": features.get("qrs_duration", 0.0),
            "rr_mean": features.get("rr_mean", 0.0),
        }
        
        return AnalysisResponse(
            prediction=ARRHYTHMIA_CLASSES[predicted_class_idx],
            confidence=confidence,
            features=key_features,
            signal_quality=quality
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["reload"]
    )

