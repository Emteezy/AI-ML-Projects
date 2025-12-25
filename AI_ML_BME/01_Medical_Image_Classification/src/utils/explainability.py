"""
Explainability utilities using Grad-CAM for model interpretability.
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import io
import base64

from ..models.resnet_model import MedicalResNet


def generate_gradcam(
    model: MedicalResNet,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for model explainability.
    
    Args:
        model: Trained model
        input_tensor: Input image tensor (1, 3, H, W)
        target_class: Target class index (None for predicted class)
        device: Device to run on
        
    Returns:
        Grad-CAM heatmap as numpy array
    """
    model.eval()
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_()
    
    # Forward pass
    output = model(input_tensor)
    
    # Get target class
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Backward pass
    model.zero_grad()
    output[0, target_class].backward()
    
    # Get gradients
    gradients = input_tensor.grad.data
    
    # Get activations from the last convolutional layer
    # We need to hook into the backbone
    activations = None
    
    def hook_fn(module, input, output):
        nonlocal activations
        activations = output
    
    # Register hook on the last conv layer of backbone
    # ResNet structure: backbone -> Sequential -> last layer is conv
    hook = None
    for name, module in model.backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hook = module.register_forward_hook(hook_fn)
    
    # Forward pass again to get activations
    with torch.no_grad():
        _ = model(input_tensor)
    
    if activations is None:
        raise ValueError("Could not extract activations")
    
    # Compute Grad-CAM
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = F.relu(cam)
    
    # Normalize
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # Resize to input size
    cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
    
    if hook is not None:
        hook.remove()
    
    return cam


def visualize_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        image: Original image (H, W, 3)
        heatmap: Grad-CAM heatmap (H, W)
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap
        
    Returns:
        Overlaid image
    """
    # Normalize image to 0-255
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Apply colormap to heatmap
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    
    # Resize heatmap to match image
    if heatmap_colored.shape[:2] != image.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))
    
    # Overlay
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def create_explanation_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    prediction: str,
    confidence: float,
) -> str:
    """
    Create a visualization with original image, heatmap, and prediction.
    
    Args:
        image: Original image
        heatmap: Grad-CAM heatmap
        prediction: Prediction class
        confidence: Prediction confidence
        
    Returns:
        Base64 encoded image string
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Heatmap
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    
    # Overlay
    overlay = visualize_gradcam(image, heatmap)
    axes[2].imshow(overlay)
    axes[2].set_title(f"Prediction: {prediction}\nConfidence: {confidence:.2%}")
    axes[2].axis("off")
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close()
    
    return img_base64

