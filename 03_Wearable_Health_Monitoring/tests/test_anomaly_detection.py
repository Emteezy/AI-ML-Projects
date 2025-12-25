"""Tests for anomaly detection."""

import pytest
from src.processing.anomaly_detection import AnomalyDetector, detect_sudden_changes, detect_outliers_iqr
import numpy as np


def test_anomaly_detector_thresholds():
    """Test anomaly detection using thresholds."""
    detector = AnomalyDetector()
    
    # Normal heart rate
    reading = {
        "sensor_type": "heart_rate",
        "heart_rate": 75,
        "timestamp": 0,
    }
    is_anomaly, severity, message = detector.detect_using_thresholds(reading)
    assert not is_anomaly or severity == "normal"
    
    # High heart rate (warning)
    reading = {
        "sensor_type": "heart_rate",
        "heart_rate": 110,
        "timestamp": 0,
    }
    is_anomaly, severity, message = detector.detect_using_thresholds(reading)
    assert is_anomaly
    assert severity in ["warning", "critical"]
    
    # Low SpO2 (critical)
    reading = {
        "sensor_type": "pulse_oximeter",
        "spo2": 85,
        "timestamp": 0,
    }
    is_anomaly, severity, message = detector.detect_using_thresholds(reading)
    assert is_anomaly
    assert severity == "critical"


def test_anomaly_detector_statistics():
    """Test anomaly detection using statistics."""
    detector = AnomalyDetector()
    
    # Calibrate with normal data
    sensor_data = [
        {"sensor_type": "heart_rate", "heart_rate": 70 + i % 10}
        for i in range(100)
    ]
    detector.calibrate(sensor_data, "heart_rate")
    
    # Normal reading
    reading = {"sensor_type": "heart_rate", "heart_rate": 75}
    is_anomaly, z_score = detector.detect_using_statistics(reading, "heart_rate")
    assert not is_anomaly
    
    # Anomalous reading
    reading = {"sensor_type": "heart_rate", "heart_rate": 150}
    is_anomaly, z_score = detector.detect_using_statistics(reading, "heart_rate")
    assert is_anomaly


def test_detect_sudden_changes():
    """Test sudden change detection."""
    # Normal data
    values = np.array([1, 1, 1, 1, 1])
    changes = detect_sudden_changes(values, threshold=2.0)
    assert len(changes) == 0
    
    # Data with sudden change
    values = np.array([1, 1, 1, 10, 10])
    changes = detect_sudden_changes(values, threshold=2.0)
    assert len(changes) > 0


def test_detect_outliers_iqr():
    """Test IQR-based outlier detection."""
    # Normal data
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    outliers = detect_outliers_iqr(values)
    assert len(outliers) == 0
    
    # Data with outliers
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    outliers = detect_outliers_iqr(values)
    assert len(outliers) > 0

