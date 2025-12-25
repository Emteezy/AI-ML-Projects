"""
Medical image preprocessing utilities.
"""
import torch
import numpy as np
from PIL import Image
from typing import Union, Tuple
import torchvision.transforms as transforms
from pathlib import Path

from ..config.settings import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image object
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path)
    
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    return image


def normalize_image(image: Image.Image) -> torch.Tensor:
    """
    Normalize image using ImageNet statistics.
    
    Args:
        image: PIL Image
        
    Returns:
        Normalized tensor
    """
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])
    
    return transform(image)


def preprocess_image(
    image: Union[str, Path, Image.Image, np.ndarray],
    return_tensor: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Preprocess image for model inference.
    
    Args:
        image: Image input (path, PIL Image, or numpy array)
        return_tensor: Whether to return tensor or numpy array
        
    Returns:
        Preprocessed image tensor or array
    """
    # Load image if path provided
    if isinstance(image, (str, Path)):
        image = load_image(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Normalize
    if return_tensor:
        tensor = normalize_image(image)
        return tensor.unsqueeze(0)  # Add batch dimension
    else:
        # Return numpy array without normalization (for visualization)
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        tensor = transform(image)
        return tensor.permute(1, 2, 0).numpy()


def preprocess_batch(images: list) -> torch.Tensor:
    """
    Preprocess a batch of images.
    
    Args:
        images: List of image inputs
        
    Returns:
        Batch tensor of shape (batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    """
    processed = [preprocess_image(img, return_tensor=True) for img in images]
    return torch.cat(processed, dim=0)

