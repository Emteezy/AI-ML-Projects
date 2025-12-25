"""Training script for ECG arrhythmia detection models."""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from typing import Tuple, List
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

from ..config.settings import (
    MODELS_DIR,
    RESULTS_DIR,
    TRAINING_CONFIG,
    ARRHYTHMIA_CLASSES,
    DEVICE,
)
from ..models import get_model, save_model
from .dataset import ECGDataset, load_mitdb_data, create_segments_from_records


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for signals, labels in tqdm(dataloader, desc="Training"):
        signals = signals.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """
    Validate model.
    
    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in tqdm(dataloader, desc="Validating"):
            signals = signals.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Path
) -> None:
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path
) -> None:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=ARRHYTHMIA_CLASSES,
        yticklabels=ARRHYTHMIA_CLASSES
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_model(
    model_type: str = "lstm",
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    data_dir: Path = None,
    save_model_flag: bool = True,
    model_name: str = None
) -> None:
    """
    Main training function.
    
    Args:
        model_type: Type of model ('lstm' or 'transformer')
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        data_dir: Path to MIT-BIH database
        save_model_flag: Whether to save the trained model
        model_name: Name for saved model
    """
    # Get config values
    epochs = epochs or TRAINING_CONFIG["epochs"]
    batch_size = batch_size or TRAINING_CONFIG["batch_size"]
    learning_rate = learning_rate or TRAINING_CONFIG["learning_rate"]
    
    # Set device
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    
    # Load data
    print("Loading MIT-BIH dataset...")
    records = load_mitdb_data(data_dir=data_dir)
    
    if len(records) == 0:
        raise ValueError(
            "No records found. Please ensure MIT-BIH dataset is downloaded "
            "and placed in the data/mitdb/ directory."
        )
    
    print(f"Loaded {len(records)} records")
    
    # Create segments
    print("Creating segments...")
    segments, labels = create_segments_from_records(records)
    print(f"Created {len(segments)} segments")
    
    # Split dataset
    dataset = ECGDataset(segments, labels)
    train_size = int(TRAINING_CONFIG["train_split"] * len(dataset))
    val_size = int(TRAINING_CONFIG["val_split"] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=TRAINING_CONFIG["num_workers"],
        pin_memory=TRAINING_CONFIG["pin_memory"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=TRAINING_CONFIG["num_workers"],
        pin_memory=TRAINING_CONFIG["pin_memory"]
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=TRAINING_CONFIG["num_workers"],
        pin_memory=TRAINING_CONFIG["pin_memory"]
    )
    
    # Create model
    print(f"Creating {model_type} model...")
    model = get_model(model_type=model_type)
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_model_flag:
                model_name = model_name or f"{model_type}_best.pth"
                model_path = MODELS_DIR / model_name
                save_model(
                    model,
                    model_path,
                    model_type=model_type,
                    metadata={
                        "epoch": epoch + 1,
                        "val_accuracy": val_acc,
                        "train_accuracy": train_acc,
                    }
                )
    
    # Test evaluation
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Get test predictions for confusion matrix
    model.eval()
    all_test_preds = []
    all_test_labels = []
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            outputs = model(signals)
            preds = torch.argmax(outputs, dim=1)
            all_test_preds.extend(preds.cpu().numpy())
            all_test_labels.extend(labels.numpy())
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        all_test_labels,
        all_test_preds,
        target_names=ARRHYTHMIA_CLASSES
    ))
    
    # Save plots
    print("\nSaving results...")
    plot_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        RESULTS_DIR / "training_curves.png"
    )
    plot_confusion_matrix(
        np.array(all_test_labels),
        np.array(all_test_preds),
        RESULTS_DIR / "confusion_matrix.png"
    )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ECG arrhythmia detection model")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "transformer"],
                        help="Model type")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to MIT-BIH database")
    parser.add_argument("--no-save", action="store_true", help="Don't save model")
    parser.add_argument("--model-name", type=str, default=None, help="Model name for saving")
    
    args = parser.parse_args()
    
    train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        save_model_flag=not args.no_save,
        model_name=args.model_name
    )

