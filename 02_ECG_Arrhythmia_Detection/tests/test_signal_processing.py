"""Tests for signal processing modules."""

import pytest
import numpy as np
from src.signal_processing import (
    preprocess_ecg_signal,
    normalize_signal,
    baseline_correction,
    apply_bandpass_filter,
    apply_notch_filter,
    denoise_signal,
    detect_qrs_complexes,
    extract_features,
    calculate_heart_rate,
)


@pytest.fixture
def sample_ecg_signal():
    """Generate a sample ECG signal for testing."""
    sampling_rate = 360
    duration = 10  # seconds
    t = np.linspace(0, duration, sampling_rate * duration)
    
    # Simulate ECG signal with QRS complexes
    signal = np.zeros_like(t)
    heart_rate = 70  # BPM
    rr_interval = 60.0 / heart_rate
    
    for i in range(int(duration / rr_interval)):
        qrs_time = i * rr_interval
        qrs_idx = int(qrs_time * sampling_rate)
        if qrs_idx < len(signal):
            qrs_width = int(0.1 * sampling_rate)
            qrs_signal = np.exp(-0.5 * ((t[qrs_idx:qrs_idx+qrs_width] - qrs_time) / 0.02) ** 2)
            if len(qrs_signal) == qrs_width:
                signal[qrs_idx:qrs_idx+qrs_width] += qrs_signal * 2.0
    
    # Add noise and baseline wander
    signal = signal + 0.1 * np.random.randn(len(signal))
    signal = signal + 0.5 * np.sin(2 * np.pi * 0.3 * t)  # Baseline wander
    
    return signal


def test_normalize_signal_zscore(sample_ecg_signal):
    """Test z-score normalization."""
    normalized = normalize_signal(sample_ecg_signal, method="zscore")
    
    assert np.isclose(np.mean(normalized), 0, atol=1e-6)
    assert np.isclose(np.std(normalized), 1, atol=1e-6)


def test_normalize_signal_minmax(sample_ecg_signal):
    """Test min-max normalization."""
    normalized = normalize_signal(sample_ecg_signal, method="minmax")
    
    assert np.min(normalized) >= 0
    assert np.max(normalized) <= 1


def test_baseline_correction(sample_ecg_signal):
    """Test baseline correction."""
    corrected = baseline_correction(sample_ecg_signal)
    
    assert len(corrected) == len(sample_ecg_signal)
    assert corrected.dtype == sample_ecg_signal.dtype


def test_preprocess_ecg_signal(sample_ecg_signal):
    """Test complete preprocessing pipeline."""
    processed = preprocess_ecg_signal(sample_ecg_signal)
    
    assert len(processed) == len(sample_ecg_signal)
    assert isinstance(processed, np.ndarray)


def test_apply_bandpass_filter(sample_ecg_signal):
    """Test bandpass filtering."""
    filtered = apply_bandpass_filter(sample_ecg_signal)
    
    assert len(filtered) == len(sample_ecg_signal)
    assert isinstance(filtered, np.ndarray)


def test_apply_notch_filter(sample_ecg_signal):
    """Test notch filtering."""
    filtered = apply_notch_filter(sample_ecg_signal)
    
    assert len(filtered) == len(sample_ecg_signal)
    assert isinstance(filtered, np.ndarray)


def test_denoise_signal(sample_ecg_signal):
    """Test denoising pipeline."""
    denoised = denoise_signal(sample_ecg_signal)
    
    assert len(denoised) == len(sample_ecg_signal)
    assert isinstance(denoised, np.ndarray)


def test_detect_qrs_complexes(sample_ecg_signal):
    """Test QRS complex detection."""
    peaks = detect_qrs_complexes(sample_ecg_signal)
    
    assert isinstance(peaks, np.ndarray)
    assert len(peaks) > 0
    assert all(0 <= p < len(sample_ecg_signal) for p in peaks)


def test_calculate_heart_rate(sample_ecg_signal):
    """Test heart rate calculation."""
    peaks = detect_qrs_complexes(sample_ecg_signal)
    
    if len(peaks) >= 2:
        heart_rate = calculate_heart_rate(peaks)
        assert 0 <= heart_rate <= 300  # Reasonable range for heart rate
        assert isinstance(heart_rate, float)


def test_extract_features(sample_ecg_signal):
    """Test feature extraction."""
    features = extract_features(sample_ecg_signal)
    
    assert isinstance(features, dict)
    assert "heart_rate" in features
    assert "signal_mean" in features
    assert "signal_std" in features
    assert isinstance(features["heart_rate"], float)

