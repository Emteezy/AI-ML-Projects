"""
ResNet-based model for medical image classification using transfer learning.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

from ..config.settings import NUM_CLASSES, MODEL_NAME


class MedicalResNet(nn.Module):
    """
    Medical image classification model using ResNet transfer learning.
    
    Supports ResNet18, ResNet34, and ResNet50 architectures.
    """
    
    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        """
        Initialize the MedicalResNet model.
        
        Args:
            model_name: ResNet architecture name ('resnet18', 'resnet34', 'resnet50')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained ImageNet weights
            freeze_backbone: Whether to freeze the backbone layers
        """
        super(MedicalResNet, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained ResNet model
        # Use weights parameter for newer torchvision versions
        if model_name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features = 512
        elif model_name == "resnet34":
            backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features = 512
        elif model_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}. Use 'resnet18', 'resnet34', or 'resnet50'")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Add custom classifier for medical image classification
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Classify
        output = self.classifier(features)
        
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone (for explainability).
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        return self.backbone(x)


def create_model(
    model_name: str = MODEL_NAME,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: str = "cpu",
) -> MedicalResNet:
    """
    Create and initialize a MedicalResNet model.
    
    Args:
        model_name: ResNet architecture name
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone layers
        device: Device to load model on
        
    Returns:
        Initialized MedicalResNet model
    """
    model = MedicalResNet(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
    
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing MedicalResNet model...")
    
    # Test ResNet18
    model = create_model("resnet18", num_classes=2, device="cpu")
    print(f"ResNet18 created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print("✅ Model test passed!")

