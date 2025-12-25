"""
Dataset loader for chest X-ray images.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional

from ..config.settings import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    IMAGE_MEAN,
    IMAGE_STD,
    CLASS_LABELS,
)


class ChestXRayDataset(Dataset):
    """Dataset class for chest X-ray images."""
    
    def __init__(
        self,
        data_dir: Path,
        transform: Optional[transforms.Compose] = None,
        is_training: bool = True,
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing class subdirectories
            transform: Image transformations
            is_training: Whether this is training data (for augmentations)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_training = is_training
        
        # Get all image paths
        self.image_paths = []
        self.labels = []
        
        for class_idx, class_name in enumerate(CLASS_LABELS):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpeg"):
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
                for img_path in class_dir.glob("*.jpg"):
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
                for img_path in class_dir.glob("*.png"):
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item from dataset.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (image tensor, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(
    batch_size: int = 32,
    num_workers: int = 0,  # Set to 0 for Windows compatibility
    data_augmentation: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get data loaders for train, validation, and test sets.
    
    Args:
        batch_size: Batch size
        num_workers: Number of worker processes
        data_augmentation: Whether to use data augmentation for training
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Training transforms (with augmentation)
    if data_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ])
    
    # Validation/Test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])
    
    # Create datasets
    train_dataset = ChestXRayDataset(TRAIN_DIR, transform=train_transform, is_training=True)
    val_dataset = ChestXRayDataset(VAL_DIR, transform=val_transform, is_training=False)
    test_dataset = ChestXRayDataset(TEST_DIR, transform=val_transform, is_training=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader

