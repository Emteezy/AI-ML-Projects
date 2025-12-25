"""Tests for model architectures."""

import pytest
import torch
from src.models import LSTMModel, TransformerModel, get_model, save_model, load_model
from src.config.settings import MODELS_DIR, ARRHYTHMIA_CLASSES, SIGNAL_CONFIG


@pytest.fixture
def sample_input():
    """Generate sample input for testing."""
    batch_size = 4
    seq_len = SIGNAL_CONFIG["window_size"]
    input_size = 1
    return torch.randn(batch_size, seq_len, input_size)


def test_lstm_model_forward(sample_input):
    """Test LSTM model forward pass."""
    model = LSTMModel()
    model.eval()
    
    with torch.no_grad():
        output = model(sample_input)
    
    assert output.shape == (sample_input.shape[0], len(ARRHYTHMIA_CLASSES))
    assert not torch.isnan(output).any()


def test_transformer_model_forward(sample_input):
    """Test Transformer model forward pass."""
    model = TransformerModel()
    model.eval()
    
    with torch.no_grad():
        output = model(sample_input)
    
    assert output.shape == (sample_input.shape[0], len(ARRHYTHMIA_CLASSES))
    assert not torch.isnan(output).any()


def test_get_model():
    """Test model factory function."""
    lstm_model = get_model(model_type="lstm")
    assert isinstance(lstm_model, LSTMModel)
    
    transformer_model = get_model(model_type="transformer")
    assert isinstance(transformer_model, TransformerModel)


def test_save_and_load_model(sample_input, tmp_path):
    """Test saving and loading models."""
    # Create model
    model = LSTMModel()
    model.eval()
    
    # Save model
    model_path = tmp_path / "test_model.pth"
    save_model(model, model_path, model_type="lstm")
    
    assert model_path.exists()
    
    # Load model
    loaded_model, metadata = load_model(model_path)
    
    assert isinstance(loaded_model, LSTMModel)
    assert metadata["model_type"] == "lstm"
    
    # Test forward pass
    with torch.no_grad():
        original_output = model(sample_input)
        loaded_output = loaded_model(sample_input)
    
    # Check outputs are similar (within tolerance)
    assert torch.allclose(original_output, loaded_output, atol=1e-5)

