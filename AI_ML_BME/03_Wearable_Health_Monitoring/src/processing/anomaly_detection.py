"""Anomaly detection for health sensor data."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
from src.config.settings import HEALTH_THRESHOLDS


class AnomalyDetector:
    """Detect anomalies in health sensor data."""
    
    def __init__(self, threshold_multiplier: float = 2.0):
        """
        Initialize anomaly detector.
        
        Args:
            threshold_multiplier: Multiplier for standard deviation threshold
        """
        self.threshold_multiplier = threshold_multiplier
        self.baseline_mean = {}
        self.baseline_std = {}
        self.is_calibrated = False
    
    def calibrate(self, sensor_data: List[Dict[str, Any]], 
                 sensor_type: str) -> None:
        """
        Calibrate detector with baseline data.
        
        Args:
            sensor_data: List of sensor readings for calibration
            sensor_type: Type of sensor ('heart_rate', 'pulse_oximeter', etc.)
        """
        if len(sensor_data) == 0:
            return
        
        # Extract values
        if sensor_type == "heart_rate":
            values = np.array([r["heart_rate"] for r in sensor_data])
        elif sensor_type == "pulse_oximeter":
            values = np.array([r["spo2"] for r in sensor_data])
        elif sensor_type == "accelerometer":
            values = np.array([r["acceleration"]["magnitude"] for r in sensor_data])
        else:
            return
        
        self.baseline_mean[sensor_type] = np.mean(values)
        self.baseline_std[sensor_type] = np.std(values)
        self.is_calibrated = True
    
    def detect_using_thresholds(self, reading: Dict[str, Any]) -> Tuple[bool, str, str]:
        """
        Detect anomalies using predefined health thresholds.
        
        Args:
            reading: Single sensor reading
        
        Returns:
            Tuple of (is_anomaly, severity, message)
        """
        sensor_type = reading.get("sensor_type", "")
        thresholds = HEALTH_THRESHOLDS.get(sensor_type, {})
        
        if sensor_type == "heart_rate":
            hr = reading.get("heart_rate", 0)
            min_normal = thresholds.get("min_normal", 60)
            max_normal = thresholds.get("max_normal", 100)
            min_warning = thresholds.get("min_warning", 50)
            max_warning = thresholds.get("max_warning", 120)
            
            if hr < min_normal or hr > max_normal:
                if hr < min_warning or hr > max_warning:
                    return True, "critical", f"Heart rate critical: {hr} bpm"
                return True, "warning", f"Heart rate abnormal: {hr} bpm"
        
        elif sensor_type == "pulse_oximeter":
            spo2 = reading.get("spo2", 0)
            min_normal = thresholds.get("min_normal", 95)
            min_warning = thresholds.get("min_warning", 90)
            min_critical = thresholds.get("min_critical", 85)
            
            if spo2 < min_normal:
                if spo2 < min_critical:
                    return True, "critical", f"SpO2 critical: {spo2}%"
                elif spo2 < min_warning:
                    return True, "warning", f"SpO2 low: {spo2}%"
                return True, "warning", f"SpO2 slightly low: {spo2}%"
        
        elif sensor_type == "temperature":
            temp = reading.get("temperature", 0)
            min_normal = thresholds.get("min_normal", 36.0)
            max_normal = thresholds.get("max_normal", 37.5)
            min_warning = thresholds.get("min_warning", 35.0)
            max_warning = thresholds.get("max_warning", 38.5)
            
            if temp < min_normal or temp > max_normal:
                if temp < min_warning or temp > max_warning:
                    return True, "critical", f"Temperature critical: {temp}°C"
                return True, "warning", f"Temperature abnormal: {temp}°C"
        
        return False, "normal", "Values within normal range"
    
    def detect_using_statistics(self, reading: Dict[str, Any], 
                               sensor_type: str) -> Tuple[bool, float]:
        """
        Detect anomalies using statistical methods (Z-score).
        
        Args:
            reading: Single sensor reading
            sensor_type: Type of sensor
        
        Returns:
            Tuple of (is_anomaly, z_score)
        """
        if not self.is_calibrated or sensor_type not in self.baseline_mean:
            return False, 0.0
        
        # Extract value
        if sensor_type == "heart_rate":
            value = reading.get("heart_rate", 0)
        elif sensor_type == "pulse_oximeter":
            value = reading.get("spo2", 0)
        elif sensor_type == "accelerometer":
            value = reading.get("acceleration", {}).get("magnitude", 0)
        else:
            return False, 0.0
        
        mean = self.baseline_mean[sensor_type]
        std = self.baseline_std[sensor_type]
        
        if std == 0:
            return False, 0.0
        
        z_score = abs((value - mean) / std)
        is_anomaly = z_score > self.threshold_multiplier
        
        return is_anomaly, z_score
    
    def detect_in_window(self, sensor_data: List[Dict[str, Any]], 
                        sensor_type: str, window_size: int = 60) -> List[Dict[str, Any]]:
        """
        Detect anomalies in a window of sensor data.
        
        Args:
            sensor_data: List of sensor readings
            sensor_type: Type of sensor
            window_size: Size of analysis window
        
        Returns:
            List of anomaly detections
        """
        anomalies = []
        window = sensor_data[-window_size:] if len(sensor_data) > window_size else sensor_data
        
        for reading in window:
            # Check thresholds
            is_anomaly_th, severity, message = self.detect_using_thresholds(reading)
            
            # Check statistics if calibrated
            is_anomaly_stat, z_score = self.detect_using_statistics(reading, sensor_type)
            
            if is_anomaly_th or is_anomaly_stat:
                anomalies.append({
                    "timestamp": reading.get("timestamp"),
                    "sensor_type": sensor_type,
                    "severity": severity,
                    "message": message,
                    "z_score": z_score,
                    "reading": reading,
                })
        
        return anomalies


def detect_sudden_changes(values: np.ndarray, threshold: float = 2.0) -> List[int]:
    """
    Detect sudden changes in time series data.
    
    Args:
        values: Array of sensor values
        threshold: Threshold for change detection (in standard deviations)
    
    Returns:
        List of indices where sudden changes occurred
    """
    if len(values) < 2:
        return []
    
    diffs = np.diff(values)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    
    if std_diff == 0:
        return []
    
    z_scores = np.abs((diffs - mean_diff) / std_diff)
    change_indices = np.where(z_scores > threshold)[0].tolist()
    
    return change_indices


def detect_outliers_iqr(values: np.ndarray) -> List[int]:
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Args:
        values: Array of sensor values
    
    Returns:
        List of indices of outliers
    """
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outlier_indices = np.where((values < lower_bound) | (values > upper_bound))[0].tolist()
    
    return outlier_indices

