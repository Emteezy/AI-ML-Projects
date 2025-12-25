"""ECG signal filtering and denoising functions."""

import numpy as np
from scipy import signal
from typing import Optional

from ..config.settings import FILTER_CONFIG


def apply_bandpass_filter(
    ecg_signal: np.ndarray,
    low_freq: float = None,
    high_freq: float = None,
    sampling_rate: int = None,
    order: int = 4
) -> np.ndarray:
    """
    Apply bandpass filter to ECG signal.
    
    Args:
        ecg_signal: Input ECG signal
        low_freq: Low cutoff frequency in Hz
        high_freq: High cutoff frequency in Hz
        sampling_rate: Sampling rate in Hz
        order: Filter order
        
    Returns:
        Filtered ECG signal
    """
    if low_freq is None:
        low_freq = FILTER_CONFIG["bandpass_low"]
    if high_freq is None:
        high_freq = FILTER_CONFIG["bandpass_high"]
    if sampling_rate is None:
        from ..config.settings import SIGNAL_CONFIG
        sampling_rate = SIGNAL_CONFIG["sampling_rate"]
    
    # Design Butterworth bandpass filter
    sos = signal.butter(
        order,
        [low_freq, high_freq],
        btype='band',
        fs=sampling_rate,
        output='sos'
    )
    
    # Apply filter (forward-backward for zero phase)
    filtered_signal = signal.sosfiltfilt(sos, ecg_signal)
    
    return filtered_signal


def apply_notch_filter(
    ecg_signal: np.ndarray,
    freq: float = None,
    sampling_rate: int = None,
    q: float = None
) -> np.ndarray:
    """
    Apply notch filter to remove powerline interference.
    
    Args:
        ecg_signal: Input ECG signal
        freq: Notch frequency in Hz (typically 50 or 60)
        sampling_rate: Sampling rate in Hz
        q: Quality factor
        
    Returns:
        Filtered ECG signal
    """
    if freq is None:
        freq = FILTER_CONFIG["notch_freq"]
    if q is None:
        q = FILTER_CONFIG["notch_q"]
    if sampling_rate is None:
        from ..config.settings import SIGNAL_CONFIG
        sampling_rate = SIGNAL_CONFIG["sampling_rate"]
    
    # Design notch filter
    b, a = signal.iirnotch(freq, q, sampling_rate)
    
    # Apply filter (forward-backward for zero phase)
    filtered_signal = signal.filtfilt(b, a, ecg_signal)
    
    return filtered_signal


def denoise_signal(
    ecg_signal: np.ndarray,
    sampling_rate: int = None,
    apply_bandpass: bool = True,
    apply_notch: bool = True
) -> np.ndarray:
    """
    Complete denoising pipeline for ECG signal.
    
    Args:
        ecg_signal: Input ECG signal
        sampling_rate: Sampling rate in Hz
        apply_bandpass: Whether to apply bandpass filter
        apply_notch: Whether to apply notch filter
        
    Returns:
        Denoised ECG signal
    """
    if sampling_rate is None:
        from ..config.settings import SIGNAL_CONFIG
        sampling_rate = SIGNAL_CONFIG["sampling_rate"]
    
    denoised_signal = ecg_signal.copy().astype(float)
    
    # Apply bandpass filter
    if apply_bandpass:
        denoised_signal = apply_bandpass_filter(denoised_signal, sampling_rate=sampling_rate)
    
    # Apply notch filter for powerline noise
    if apply_notch:
        denoised_signal = apply_notch_filter(denoised_signal, sampling_rate=sampling_rate)
    
    return denoised_signal

