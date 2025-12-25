"""ECG signal preprocessing functions."""

import numpy as np
from scipy import signal
from typing import List, Tuple

from ..config.settings import SIGNAL_CONFIG


def baseline_correction(ecg_signal: np.ndarray, sampling_rate: int = None) -> np.ndarray:
    """
    Remove baseline wander from ECG signal using high-pass filter.
    
    Args:
        ecg_signal: Input ECG signal
        sampling_rate: Sampling rate in Hz (default from config)
        
    Returns:
        Baseline-corrected ECG signal
    """
    if sampling_rate is None:
        sampling_rate = SIGNAL_CONFIG["sampling_rate"]
    
    # High-pass filter at 0.5 Hz to remove baseline wander
    sos = signal.butter(4, 0.5, btype='high', fs=sampling_rate, output='sos')
    corrected_signal = signal.sosfiltfilt(sos, ecg_signal)
    
    return corrected_signal


def normalize_signal(ecg_signal: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize ECG signal.
    
    Args:
        ecg_signal: Input ECG signal
        method: Normalization method ('zscore', 'minmax', 'unit')
        
    Returns:
        Normalized ECG signal
    """
    if method == "zscore":
        mean = np.mean(ecg_signal)
        std = np.std(ecg_signal)
        if std == 0:
            return ecg_signal - mean
        return (ecg_signal - mean) / std
    
    elif method == "minmax":
        min_val = np.min(ecg_signal)
        max_val = np.max(ecg_signal)
        if max_val == min_val:
            return np.zeros_like(ecg_signal)
        return (ecg_signal - min_val) / (max_val - min_val)
    
    elif method == "unit":
        norm = np.linalg.norm(ecg_signal)
        if norm == 0:
            return ecg_signal
        return ecg_signal / norm
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def segment_signal(
    ecg_signal: np.ndarray,
    window_size: int = None,
    overlap: float = None,
    sampling_rate: int = None
) -> List[np.ndarray]:
    """
    Segment ECG signal into windows.
    
    Args:
        ecg_signal: Input ECG signal
        window_size: Size of each window in samples
        overlap: Overlap ratio between windows (0-1)
        sampling_rate: Sampling rate in Hz
        
    Returns:
        List of signal segments
    """
    if window_size is None:
        window_size = SIGNAL_CONFIG["window_size"]
    if overlap is None:
        overlap = SIGNAL_CONFIG["overlap"]
    
    segments = []
    step_size = int(window_size * (1 - overlap))
    
    for start in range(0, len(ecg_signal) - window_size + 1, step_size):
        end = start + window_size
        segments.append(ecg_signal[start:end])
    
    return segments


def preprocess_ecg_signal(
    ecg_signal: np.ndarray,
    sampling_rate: int = None,
    normalize: bool = True,
    correct_baseline: bool = True
) -> np.ndarray:
    """
    Complete preprocessing pipeline for ECG signal.
    
    Args:
        ecg_signal: Raw ECG signal
        sampling_rate: Sampling rate in Hz
        normalize: Whether to normalize the signal
        correct_baseline: Whether to correct baseline wander
        
    Returns:
        Preprocessed ECG signal
    """
    if sampling_rate is None:
        sampling_rate = SIGNAL_CONFIG["sampling_rate"]
    
    processed_signal = ecg_signal.copy().astype(float)
    
    # Baseline correction
    if correct_baseline:
        processed_signal = baseline_correction(processed_signal, sampling_rate)
    
    # Normalization
    if normalize:
        processed_signal = normalize_signal(processed_signal, method="zscore")
    
    return processed_signal

