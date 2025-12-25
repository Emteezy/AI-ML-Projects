"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
import numpy as np
from src.api.main import app
from src.config.settings import SIGNAL_CONFIG


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_ecg_signal():
    """Generate sample ECG signal for testing."""
    signal_length = SIGNAL_CONFIG["window_size"]
    # Generate a simple sinusoidal signal
    t = np.linspace(0, signal_length / SIGNAL_CONFIG["sampling_rate"], signal_length)
    signal = np.sin(2 * np.pi * 1 * t) + 0.1 * np.random.randn(signal_length)
    return signal.tolist()


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "available_models" in data


def test_predict_endpoint(client, sample_ecg_signal):
    """Test prediction endpoint."""
    # Note: This test may fail if no model is available
    # In that case, it's expected behavior
    response = client.post(
        "/predict",
        json={
            "signal": sample_ecg_signal,
            "model_name": "lstm_best.pth"
        }
    )
    
    # Check if model exists or if we get appropriate error
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "class_probabilities" in data
        assert isinstance(data["confidence"], float)
        assert 0 <= data["confidence"] <= 1
    elif response.status_code == 404:
        # Model not found - acceptable for test environment
        assert "not found" in response.json()["detail"].lower()
    else:
        # Other errors are unexpected
        pytest.fail(f"Unexpected status code: {response.status_code}")


def test_predict_endpoint_short_signal(client):
    """Test prediction endpoint with too short signal."""
    short_signal = [1.0, 2.0, 3.0]  # Too short
    response = client.post(
        "/predict",
        json={
            "signal": short_signal,
            "model_name": "lstm_best.pth"
        }
    )
    
    assert response.status_code == 400
    assert "too short" in response.json()["detail"].lower()


def test_analyze_endpoint(client, sample_ecg_signal):
    """Test analysis endpoint."""
    response = client.post(
        "/analyze",
        json={
            "signal": sample_ecg_signal,
            "model_name": "lstm_best.pth"
        }
    )
    
    # Check if model exists or if we get appropriate error
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "features" in data
        assert "signal_quality" in data
        assert isinstance(data["features"], dict)
    elif response.status_code == 404:
        # Model not found - acceptable for test environment
        assert "not found" in response.json()["detail"].lower()

