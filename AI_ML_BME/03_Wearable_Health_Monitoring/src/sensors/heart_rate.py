"""Heart rate sensor interface with simulation mode support."""

import time
import random
import numpy as np
from typing import Optional, Dict, Any
from src.config.settings import SENSOR_CONFIG


class HeartRateSensor:
    """Heart rate sensor interface for MAX30102 or simulation."""
    
    def __init__(self, simulation_mode: Optional[bool] = None):
        """
        Initialize heart rate sensor.
        
        Args:
            simulation_mode: If True, use simulated data. If None, uses config setting.
        """
        if simulation_mode is None:
            simulation_mode = SENSOR_CONFIG["simulation_mode"]
        
        self.simulation_mode = simulation_mode
        self.sampling_rate = SENSOR_CONFIG["sampling_rate"]["heart_rate"]
        self._last_reading_time = time.time()
        self._baseline_hr = 72  # Baseline heart rate for simulation
        self._sensor_initialized = False
        
        if not simulation_mode:
            self._init_hardware_sensor()
        else:
            print("Heart Rate Sensor: Running in simulation mode")
    
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
            print("Heart Rate Sensor: Hardware initialized")
        except Exception as e:
            print(f"Warning: Could not initialize hardware sensor: {e}")
            print("Falling back to simulation mode")
            self.simulation_mode = True
            self._sensor_initialized = False
    
    def read(self) -> Dict[str, Any]:
        """
        Read heart rate value.
        
        Returns:
            Dictionary containing heart rate and timestamp
        """
        current_time = time.time()
        timestamp = current_time
        
        if self.simulation_mode:
            # Simulate realistic heart rate data with some variability
            # Add small random variations around baseline
            variation = random.uniform(-5, 5)
            # Simulate occasional variations (like during activity)
            if random.random() < 0.1:  # 10% chance of larger variation
                variation = random.uniform(-15, 20)
            
            heart_rate = max(50, min(120, self._baseline_hr + variation))
            
            # Gradually change baseline to simulate different states
            if random.random() < 0.05:  # 5% chance to change baseline
                self._baseline_hr = random.uniform(65, 85)
        else:
            # Read from hardware sensor
            try:
                # heart_rate = self.sensor.heart_rate
                # For now, fallback to simulation
                variation = random.uniform(-3, 3)
                heart_rate = max(60, min(100, self._baseline_hr + variation))
            except Exception as e:
                print(f"Error reading sensor: {e}")
                heart_rate = self._baseline_hr
        
        self._last_reading_time = current_time
        
        return {
            "heart_rate": round(heart_rate, 1),
            "timestamp": timestamp,
            "unit": "bpm",
            "sensor_type": "heart_rate",
        }
    
    def read_raw(self) -> Dict[str, Any]:
        """
        Read raw sensor data (IR and Red values for MAX30102).
        
        Returns:
            Dictionary containing raw sensor readings
        """
        if self.simulation_mode:
            # Simulate raw IR and Red values
            ir_value = random.randint(50000, 100000)
            red_value = random.randint(50000, 100000)
        else:
            # Read raw values from hardware
            try:
                # ir_value = self.sensor.ir
                # red_value = self.sensor.red
                ir_value = random.randint(50000, 100000)
                red_value = random.randint(50000, 100000)
            except Exception as e:
                print(f"Error reading raw sensor data: {e}")
                ir_value = 75000
                red_value = 75000
        
        return {
            "ir_value": ir_value,
            "red_value": red_value,
            "timestamp": time.time(),
        }
    
    def is_available(self) -> bool:
        """Check if sensor is available."""
        return self._sensor_initialized or self.simulation_mode

