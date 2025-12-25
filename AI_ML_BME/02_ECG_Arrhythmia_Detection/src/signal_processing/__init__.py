"""Signal processing modules for ECG analysis."""

from .filtering import apply_bandpass_filter, apply_notch_filter, denoise_signal
from .preprocessing import (
    baseline_correction,
    normalize_signal,
    preprocess_ecg_signal,
    segment_signal,
)
from .features import (
    detect_qrs_complexes,
    extract_features,
    calculate_heart_rate,
    calculate_rr_intervals,
)

__all__ = [
    "preprocess_ecg_signal",
    "normalize_signal",
    "baseline_correction",
    "segment_signal",
    "apply_bandpass_filter",
    "apply_notch_filter",
    "denoise_signal",
    "detect_qrs_complexes",
    "extract_features",
    "calculate_heart_rate",
    "calculate_rr_intervals",
]

