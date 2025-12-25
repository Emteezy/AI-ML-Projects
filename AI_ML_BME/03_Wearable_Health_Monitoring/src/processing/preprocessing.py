"""Data preprocessing utilities for health sensor data."""

import numpy as np
from typing import List, Dict, Any, Optional
from scipy import signal


def normalize_data(data: np.ndarray, method: str = "min_max") -> np.ndarray:
    """
    Normalize sensor data.
    
    Args:
        data: Input data array
        method: Normalization method ('min_max' or 'z_score')
    
    Returns:
        Normalized data array
    """
    if method == "min_max":
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min == 0:
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)
    elif method == "z_score":
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return np.zeros_like(data)
        return (data - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def filter_noise(data: np.ndarray, filter_type: str = "butter", 
                 cutoff: float = 0.1, order: int = 4) -> np.ndarray:
    """
    Apply noise filter to sensor data.
    
    Args:
        data: Input data array
        filter_type: Type of filter ('butter', 'median', or 'moving_average')
        cutoff: Cutoff frequency for butterworth filter
        order: Filter order
    
    Returns:
        Filtered data array
    """
    if filter_type == "butter":
        b, a = signal.butter(order, cutoff, btype='low')
        return signal.filtfilt(b, a, data)
    elif filter_type == "median":
        return signal.medfilt(data, kernel_size=5)
    elif filter_type == "moving_average":
        window_size = 5
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def extract_features(sensor_data: List[Dict[str, Any]], 
                    window_size: int = 60) -> Dict[str, float]:
    """
    Extract statistical features from sensor data window.
    
    Args:
        sensor_data: List of sensor readings
        window_size: Size of the window for feature extraction
    
    Returns:
        Dictionary of extracted features
    """
    if len(sensor_data) == 0:
        return {}
    
    # Use the last window_size readings or all available
    window = sensor_data[-window_size:] if len(sensor_data) > window_size else sensor_data
    
    # Extract values based on sensor type
    sensor_type = window[0].get("sensor_type", "")
    
    if sensor_type == "heart_rate":
        values = np.array([r["heart_rate"] for r in window])
    elif sensor_type == "pulse_oximeter":
        values = np.array([r["spo2"] for r in window])
    elif sensor_type == "accelerometer":
        values = np.array([r["acceleration"]["magnitude"] for r in window])
    else:
        return {}
    
    features = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "range": float(np.max(values) - np.min(values)),
    }
    
    # Additional features for time series
    if len(values) > 1:
        # Rate of change
        features["rate_of_change"] = float(np.mean(np.diff(values)))
        # Variance
        features["variance"] = float(np.var(values))
    
    return features


def resample_data(data: np.ndarray, original_rate: float, 
                 target_rate: float) -> np.ndarray:
    """
    Resample data to target sampling rate.
    
    Args:
        data: Input data array
        original_rate: Original sampling rate in Hz
        target_rate: Target sampling rate in Hz
    
    Returns:
        Resampled data array
    """
    if original_rate == target_rate:
        return data
    
    num_samples = int(len(data) * target_rate / original_rate)
    return signal.resample(data, num_samples)


def create_sequences(data: List[Dict[str, Any]], 
                    sequence_length: int = 30) -> List[np.ndarray]:
    """
    Create sequences from sensor data for ML model input.
    
    Args:
        data: List of sensor readings
        sequence_length: Length of each sequence
    
    Returns:
        List of sequences (arrays)
    """
    sequences = []
    
    # Extract values
    sensor_type = data[0].get("sensor_type", "")
    
    if sensor_type == "heart_rate":
        values = np.array([r["heart_rate"] for r in data])
    elif sensor_type == "pulse_oximeter":
        values = np.array([r["spo2"] for r in data])
    elif sensor_type == "accelerometer":
        values = np.array([r["acceleration"]["magnitude"] for r in data])
    else:
        return []
    
    # Create overlapping sequences
    for i in range(len(values) - sequence_length + 1):
        sequences.append(values[i:i + sequence_length])
    
    return sequences if sequences else [np.array(values)]


def smooth_data(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Smooth data using moving average.
    
    Args:
        data: Input data array
        window_size: Size of smoothing window
    
    Returns:
        Smoothed data array
    """
    if len(data) < window_size:
        return data
    
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='same')

