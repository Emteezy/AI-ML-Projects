"""
Utility functions for model loading, saving, and management.
"""
import torch
from pathlib import Path
from typing import Dict, Optional, Any
import json

from .resnet_model import MedicalResNet, create_model
from ..config.settings import MODEL_PATH, MODEL_NAME, NUM_CLASSES, DEVICE


def save_model(
    model: MedicalResNet,
    save_path: str,
    epoch: Optional[int] = None,
    accuracy: Optional[float] = None,
    loss: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model checkpoint with metadata.
    
    Args:
        model: Model to save
        save_path: Path to save the model
        epoch: Training epoch number
        accuracy: Model accuracy
        loss: Model loss
        metadata: Additional metadata to save
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": model.model_name,
        "num_classes": model.num_classes,
        "epoch": epoch,
        "accuracy": accuracy,
        "loss": loss,
        "metadata": metadata or {},
    }
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def load_model(
    model_path: str,
    device: str = DEVICE,
    model_name: Optional[str] = None,
) -> tuple[MedicalResNet, Dict[str, Any]]:
    """
    Load model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        model_name: Model architecture name (if not in checkpoint)
        
    Returns:
        Tuple of (model, metadata)
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model name from checkpoint or parameter
    model_name = checkpoint.get("model_name", model_name or MODEL_NAME)
    num_classes = checkpoint.get("num_classes", NUM_CLASSES)
    
    # Create model
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,  # We're loading trained weights
        device=device,
    )
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Extract metadata
    metadata = {
        "epoch": checkpoint.get("epoch"),
        "accuracy": checkpoint.get("accuracy"),
        "loss": checkpoint.get("loss"),
        "model_name": model_name,
        "num_classes": num_classes,
        **checkpoint.get("metadata", {}),
    }
    
    print(f"Model loaded from {model_path}")
    print(f"Model: {model_name}, Epoch: {metadata.get('epoch')}, Accuracy: {metadata.get('accuracy')}")
    
    return model, metadata


def get_model_info(model: MedicalResNet) -> Dict[str, Any]:
    """
    Get information about the model.
    
    Args:
        model: Model to get info for
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_name": model.model_name,
        "num_classes": model.num_classes,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
    }

