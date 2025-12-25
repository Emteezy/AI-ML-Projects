"""LSTM model for ECG arrhythmia classification."""

import torch
import torch.nn as nn
from typing import Dict

from ..config.settings import MODEL_CONFIG


class LSTMModel(nn.Module):
    """
    LSTM-based model for ECG arrhythmia classification.
    
    Architecture:
        - LSTM layers for sequential pattern recognition
        - Fully connected layers for classification
        - Dropout for regularization
    """
    
    def __init__(
        self,
        input_size: int = None,
        hidden_size: int = None,
        num_layers: int = None,
        dropout: float = None,
        num_classes: int = None,
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Size of input features (1 for single-channel ECG)
            hidden_size: Size of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            num_classes: Number of arrhythmia classes
        """
        super(LSTMModel, self).__init__()
        
        # Get default values from config
        config = MODEL_CONFIG["lstm"]
        input_size = input_size or config["input_size"]
        hidden_size = hidden_size or config["hidden_size"]
        num_layers = num_layers or config["num_layers"]
        dropout = dropout if dropout is not None else config["dropout"]
        num_classes = num_classes or config["num_classes"]
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class BidirectionalLSTMModel(nn.Module):
    """
    Bidirectional LSTM model for ECG arrhythmia classification.
    Uses both forward and backward context.
    """
    
    def __init__(
        self,
        input_size: int = None,
        hidden_size: int = None,
        num_layers: int = None,
        dropout: float = None,
        num_classes: int = None,
    ):
        """
        Initialize bidirectional LSTM model.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            num_classes: Number of arrhythmia classes
        """
        super(BidirectionalLSTMModel, self).__init__()
        
        config = MODEL_CONFIG["lstm"]
        input_size = input_size or config["input_size"]
        hidden_size = hidden_size or config["hidden_size"]
        num_layers = num_layers or config["num_layers"]
        dropout = dropout if dropout is not None else config["dropout"]
        num_classes = num_classes or config["num_classes"]
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Fully connected layers (hidden_size * 2 because bidirectional)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

