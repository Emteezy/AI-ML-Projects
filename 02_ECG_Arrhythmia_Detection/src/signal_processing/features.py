"""Feature extraction from ECG signals."""

import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional

from ..config.settings import SIGNAL_CONFIG


def detect_qrs_complexes(
    ecg_signal: np.ndarray,
    sampling_rate: int = None
) -> np.ndarray:
    """
    Detect QRS complexes in ECG signal using Pan-Tompkins algorithm.
    
    Args:
        ecg_signal: Input ECG signal
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Array of QRS peak indices
    """
    if sampling_rate is None:
        sampling_rate = SIGNAL_CONFIG["sampling_rate"]
    
    # Pan-Tompkins algorithm (simplified version)
    # 1. Bandpass filter (5-15 Hz) to emphasize QRS
    sos = signal.butter(2, [5, 15], btype='band', fs=sampling_rate, output='sos')
    filtered = signal.sosfiltfilt(sos, ecg_signal)
    
    # 2. Derivative to emphasize sharp changes
    derivative = np.diff(filtered)
    
    # 3. Squaring to make all values positive
    squared = derivative ** 2
    
    # 4. Moving average integration
    window_size = int(0.15 * sampling_rate)  # 150ms window
    integrated = np.convolve(squared, np.ones(window_size) / window_size, mode='same')
    
    # 5. Find peaks
    # Minimum distance between peaks (RR interval for 200 bpm max)
    min_distance = int(0.3 * sampling_rate)  # 300ms = 200 bpm
    
    # Threshold as percentage of max
    threshold = 0.3 * np.max(integrated)
    
    peaks, _ = signal.find_peaks(integrated, distance=min_distance, height=threshold)
    
    return peaks


def calculate_rr_intervals(qrs_peaks: np.ndarray, sampling_rate: int = None) -> np.ndarray:
    """
    Calculate RR intervals from QRS peaks.
    
    Args:
        qrs_peaks: Indices of QRS peaks
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Array of RR intervals in seconds
    """
    if sampling_rate is None:
        sampling_rate = SIGNAL_CONFIG["sampling_rate"]
    
    if len(qrs_peaks) < 2:
        return np.array([])
    
    # Calculate intervals between consecutive peaks
    rr_intervals = np.diff(qrs_peaks) / sampling_rate
    
    return rr_intervals


def calculate_heart_rate(qrs_peaks: np.ndarray, sampling_rate: int = None) -> float:
    """
    Calculate average heart rate from QRS peaks.
    
    Args:
        qrs_peaks: Indices of QRS peaks
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Average heart rate in BPM
    """
    if sampling_rate is None:
        sampling_rate = SIGNAL_CONFIG["sampling_rate"]
    
    if len(qrs_peaks) < 2:
        return 0.0
    
    rr_intervals = calculate_rr_intervals(qrs_peaks, sampling_rate)
    
    if len(rr_intervals) == 0:
        return 0.0
    
    # Average RR interval
    avg_rr = np.mean(rr_intervals)
    
    # Heart rate in BPM
    heart_rate = 60.0 / avg_rr if avg_rr > 0 else 0.0
    
    return heart_rate


def extract_features(
    ecg_signal: np.ndarray,
    sampling_rate: int = None
) -> Dict[str, float]:
    """
    Extract comprehensive features from ECG signal.
    
    Args:
        ecg_signal: Input ECG signal
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary of extracted features
    """
    if sampling_rate is None:
        sampling_rate = SIGNAL_CONFIG["sampling_rate"]
    
    features = {}
    
    # Detect QRS complexes
    qrs_peaks = detect_qrs_complexes(ecg_signal, sampling_rate)
    
    # Heart rate
    features["heart_rate"] = calculate_heart_rate(qrs_peaks, sampling_rate)
    
    # RR intervals
    rr_intervals = calculate_rr_intervals(qrs_peaks, sampling_rate)
    if len(rr_intervals) > 0:
        features["rr_mean"] = float(np.mean(rr_intervals))
        features["rr_std"] = float(np.std(rr_intervals))
        features["rr_min"] = float(np.min(rr_intervals))
        features["rr_max"] = float(np.max(rr_intervals))
    else:
        features["rr_mean"] = 0.0
        features["rr_std"] = 0.0
        features["rr_min"] = 0.0
        features["rr_max"] = 0.0
    
    # QRS duration (approximate, based on peak width)
    if len(qrs_peaks) > 0:
        # Find peak width at half height
        peak_widths = signal.peak_widths(
            ecg_signal, qrs_peaks, rel_height=0.5
        )[0]
        features["qrs_duration"] = float(np.mean(peak_widths) / sampling_rate)
    else:
        features["qrs_duration"] = 0.0
    
    # Signal statistics
    features["signal_mean"] = float(np.mean(ecg_signal))
    features["signal_std"] = float(np.std(ecg_signal))
    features["signal_max"] = float(np.max(ecg_signal))
    features["signal_min"] = float(np.min(ecg_signal))
    features["signal_range"] = float(np.max(ecg_signal) - np.min(ecg_signal))
    
    # Frequency domain features
    fft_vals = np.fft.rfft(ecg_signal)
    fft_freq = np.fft.rfftfreq(len(ecg_signal), 1 / sampling_rate)
    power_spectrum = np.abs(fft_vals) ** 2
    
    # Dominant frequency
    dominant_freq_idx = np.argmax(power_spectrum)
    features["dominant_frequency"] = float(fft_freq[dominant_freq_idx])
    
    # Spectral power in different bands
    low_freq_mask = (fft_freq >= 0.04) & (fft_freq <= 0.15)  # Very low frequency
    mid_freq_mask = (fft_freq > 0.15) & (fft_freq <= 0.4)    # Low frequency
    high_freq_mask = (fft_freq > 0.4) & (fft_freq <= 0.5)    # High frequency
    
    features["lf_power"] = float(np.sum(power_spectrum[mid_freq_mask]))
    features["hf_power"] = float(np.sum(power_spectrum[high_freq_mask]))
    
    return features

