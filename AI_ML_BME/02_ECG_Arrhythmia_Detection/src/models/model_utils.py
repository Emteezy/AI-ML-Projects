"""Model utility functions for loading and saving models."""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from ..config.settings import MODELS_DIR, MODEL_CONFIG, ARRHYTHMIA_CLASSES
from .lstm_model import LSTMModel, BidirectionalLSTMModel
from .transformer_model import TransformerModel


def get_model(
    model_type: str = "lstm",
    bidirectional: bool = False,
    **kwargs
) -> torch.nn.Module:
    """
    Get model instance by type.
    
    Args:
        model_type: Type of model ('lstm' or 'transformer')
        bidirectional: Whether to use bidirectional LSTM (only for LSTM)
        **kwargs: Additional model parameters
        
    Returns:
        Model instance
    """
    if model_type.lower() == "lstm":
        if bidirectional:
            model = BidirectionalLSTMModel(**kwargs)
        else:
            model = LSTMModel(**kwargs)
    elif model_type.lower() == "transformer":
        model = TransformerModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def save_model(
    model: torch.nn.Module,
    model_path: Path,
    model_type: str = "lstm",
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model to disk.
    
    Args:
        model: Model to save
        model_path: Path to save model
        model_type: Type of model
        metadata: Additional metadata to save
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare save dictionary
    save_dict = {
        "model_state_dict": model.state_dict(),
        "model_type": model_type,
        "model_config": MODEL_CONFIG.get(model_type, {}),
        "classes": ARRHYTHMIA_CLASSES,
    }
    
    if metadata:
        save_dict.update(metadata)
    
    # Save model
    torch.save(save_dict, model_path)
    print(f"Model saved to {model_path}")


def load_model(
    model_path: Path,
    device: str = "cpu",
    model_type: Optional[str] = None
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load model from disk.
    
    Args:
        model_path: Path to model file
        device: Device to load model on
        model_type: Model type (if None, inferred from saved file)
        
    Returns:
        Tuple of (model, metadata)
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model type
    if model_type is None:
        model_type = checkpoint.get("model_type", "lstm")
    
    # Get model config
    model_config = checkpoint.get("model_config", MODEL_CONFIG.get(model_type, {}))
    
    # Create model
    model = get_model(model_type=model_type, **model_config)
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Extract metadata
    metadata = {
        "model_type": checkpoint.get("model_type", model_type),
        "classes": checkpoint.get("classes", ARRHYTHMIA_CLASSES),
        **{k: v for k, v in checkpoint.items() 
           if k not in ["model_state_dict", "model_type", "model_config", "classes"]}
    }
    
    return model, metadata

