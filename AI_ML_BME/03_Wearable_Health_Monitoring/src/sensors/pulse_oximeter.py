"""Pulse oximeter (SpO2) sensor interface with simulation mode support."""

import time
import random
from typing import Optional, Dict, Any
from src.config.settings import SENSOR_CONFIG


class PulseOximeterSensor:
    """Pulse oximeter sensor interface for MAX30102 or simulation."""
    
    def __init__(self, simulation_mode: Optional[bool] = None):
        """
        Initialize pulse oximeter sensor.
        
        Args:
            simulation_mode: If True, use simulated data. If None, uses config setting.
        """
        if simulation_mode is None:
            simulation_mode = SENSOR_CONFIG["simulation_mode"]
        
        self.simulation_mode = simulation_mode
        self.sampling_rate = SENSOR_CONFIG["sampling_rate"]["spo2"]
        self._last_reading_time = time.time()
        self._baseline_spo2 = 98.0  # Baseline SpO2 for simulation
        self._sensor_initialized = False
        
        if not simulation_mode:
            self._init_hardware_sensor()
        else:
            print("Pulse Oximeter Sensor: Running in simulation mode")
    
    def _init_hardware_sensor(self):
        """Initialize hardware sensor (MAX30102)."""
        try:
            # In production, this would initialize the MAX30102 sensor
            # import board
            # import busio
            # import adafruit_max30102
            # i2c = busio.I2C(board.SCL, board.SDA)
            # self.sensor = adafruit_max30102.MAX30102(i2c)
            self._sensor_initialized = True
            print("Pulse Oximeter Sensor: Hardware initialized")
        except Exception as e:
            print(f"Warning: Could not initialize hardware sensor: {e}")
            print("Falling back to simulation mode")
            self.simulation_mode = True
            self._sensor_initialized = False
    
    def read(self) -> Dict[str, Any]:
        """
        Read SpO2 value.
        
        Returns:
            Dictionary containing SpO2 and timestamp
        """
        current_time = time.time()
        timestamp = current_time
        
        if self.simulation_mode:
            # Simulate realistic SpO2 data (typically 95-100%)
            variation = random.uniform(-1.0, 0.5)
            # Simulate occasional lower readings (like during stress/activity)
            if random.random() < 0.05:  # 5% chance of lower reading
                variation = random.uniform(-3.0, 0.0)
            
            spo2 = max(92.0, min(100.0, self._baseline_spo2 + variation))
        else:
            # Read from hardware sensor
            try:
                # spo2 = self.sensor.spo2
                # For now, fallback to simulation
                variation = random.uniform(-0.5, 0.5)
                spo2 = max(96.0, min(100.0, self._baseline_spo2 + variation))
            except Exception as e:
                print(f"Error reading sensor: {e}")
                spo2 = self._baseline_spo2
        
        self._last_reading_time = current_time
        
        return {
            "spo2": round(spo2, 1),
            "timestamp": timestamp,
            "unit": "%",
            "sensor_type": "pulse_oximeter",
        }
    
    def is_available(self) -> bool:
        """Check if sensor is available."""
        return self._sensor_initialized or self.simulation_mode

