"""Transformer model for ECG arrhythmia classification."""

import torch
import torch.nn as nn
import math
from typing import Optional

from ..config.settings import MODEL_CONFIG


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerModel(nn.Module):
    """
    Transformer-based model for ECG arrhythmia classification.
    
    Uses self-attention mechanism to capture long-range dependencies
    in ECG signals.
    """
    
    def __init__(
        self,
        input_size: int = None,
        d_model: int = None,
        nhead: int = None,
        num_layers: int = None,
        dim_feedforward: int = None,
        dropout: float = None,
        num_classes: int = None,
        max_seq_len: int = 3600,
    ):
        """
        Initialize Transformer model.
        
        Args:
            input_size: Size of input features (1 for single-channel ECG)
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            num_classes: Number of arrhythmia classes
            max_seq_len: Maximum sequence length for positional encoding
        """
        super(TransformerModel, self).__init__()
        
        # Get default values from config
        config = MODEL_CONFIG["transformer"]
        input_size = input_size or config["input_size"]
        d_model = d_model or config["d_model"]
        nhead = nhead or config["nhead"]
        num_layers = num_layers or config["num_layers"]
        dim_feedforward = dim_feedforward or config["dim_feedforward"]
        dropout = dropout if dropout is not None else config["dropout"]
        num_classes = num_classes or config["num_classes"]
        
        self.input_size = input_size
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Input projection to d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Global average pooling (or use CLS token)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to d_model
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Transpose for transformer: (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x, mask=mask)  # (seq_len, batch_size, d_model)
        
        # Transpose back: (batch_size, seq_len, d_model)
        x = x.transpose(0, 1)
        
        # Global average pooling across sequence
        # Transpose for pooling: (batch_size, d_model, seq_len)
        x = x.transpose(1, 2)
        x = self.global_pool(x)  # (batch_size, d_model, 1)
        x = x.squeeze(2)  # (batch_size, d_model)
        
        # Classification head
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

