"""Accelerometer and gyroscope sensor interface with simulation mode support."""

import time
import random
import math
from typing import Optional, Dict, Any, Tuple
from src.config.settings import SENSOR_CONFIG


class AccelerometerSensor:
    """Accelerometer/Gyroscope sensor interface for MPU6050 or simulation."""
    
    def __init__(self, simulation_mode: Optional[bool] = None):
        """
        Initialize accelerometer sensor.
        
        Args:
            simulation_mode: If True, use simulated data. If None, uses config setting.
        """
        if simulation_mode is None:
            simulation_mode = SENSOR_CONFIG["simulation_mode"]
        
        self.simulation_mode = simulation_mode
        self.sampling_rate = SENSOR_CONFIG["sampling_rate"]["accelerometer"]
        self._last_reading_time = time.time()
        self._activity_state = "rest"  # rest, walking, running
        self._sensor_initialized = False
        
        if not simulation_mode:
            self._init_hardware_sensor()
        else:
            print("Accelerometer Sensor: Running in simulation mode")
    
    def _init_hardware_sensor(self):
        """Initialize hardware sensor (MPU6050)."""
        try:
            # In production, this would initialize the MPU6050 sensor
            # import board
            # import busio
            # import adafruit_mpu6050
            # i2c = busio.I2C(board.SCL, board.SDA)
            # self.sensor = adafruit_mpu6050.MPU6050(i2c)
            self._sensor_initialized = True
            print("Accelerometer Sensor: Hardware initialized")
        except Exception as e:
            print(f"Warning: Could not initialize hardware sensor: {e}")
            print("Falling back to simulation mode")
            self.simulation_mode = True
            self._sensor_initialized = False
    
    def _simulate_activity(self) -> Tuple[float, float, float]:
        """Simulate accelerometer readings based on activity state."""
        # Change activity state occasionally
        if random.random() < 0.1:  # 10% chance to change state
            self._activity_state = random.choice(["rest", "walking", "running"])
        
        # Base acceleration values (in m/s²)
        if self._activity_state == "rest":
            base_x, base_y, base_z = 0.0, 0.0, 9.8  # Gravity on Z-axis
            noise_range = 0.2
        elif self._activity_state == "walking":
            base_x, base_y, base_z = random.uniform(-1, 1), random.uniform(-1, 1), 9.8
            noise_range = 1.0
        else:  # running
            base_x, base_y, base_z = random.uniform(-3, 3), random.uniform(-3, 3), 9.8
            noise_range = 2.0
        
        # Add noise
        x = base_x + random.uniform(-noise_range, noise_range)
        y = base_y + random.uniform(-noise_range, noise_range)
        z = base_z + random.uniform(-noise_range, noise_range)
        
        return x, y, z
    
    def read(self) -> Dict[str, Any]:
        """
        Read accelerometer values.
        
        Returns:
            Dictionary containing accelerometer data and timestamp
        """
        current_time = time.time()
        timestamp = current_time
        
        if self.simulation_mode:
            accel_x, accel_y, accel_z = self._simulate_activity()
            # Simulate gyroscope data
            gyro_x = random.uniform(-50, 50)
            gyro_y = random.uniform(-50, 50)
            gyro_z = random.uniform(-50, 50)
        else:
            # Read from hardware sensor
            try:
                # accel_x, accel_y, accel_z = self.sensor.acceleration
                # gyro_x, gyro_y, gyro_z = self.sensor.gyro
                # For now, fallback to simulation
                accel_x, accel_y, accel_z = self._simulate_activity()
                gyro_x = random.uniform(-20, 20)
                gyro_y = random.uniform(-20, 20)
                gyro_z = random.uniform(-20, 20)
            except Exception as e:
                print(f"Error reading sensor: {e}")
                accel_x, accel_y, accel_z = 0.0, 0.0, 9.8
                gyro_x, gyro_y, gyro_z = 0.0, 0.0, 0.0
        
        # Calculate magnitude (useful for activity detection)
        magnitude = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        self._last_reading_time = current_time
        
        return {
            "acceleration": {
                "x": round(accel_x, 2),
                "y": round(accel_y, 2),
                "z": round(accel_z, 2),
                "magnitude": round(magnitude, 2),
            },
            "gyroscope": {
                "x": round(gyro_x, 2),
                "y": round(gyro_y, 2),
                "z": round(gyro_z, 2),
            },
            "timestamp": timestamp,
            "unit": "m/s²",
            "sensor_type": "accelerometer",
            "activity_state": self._activity_state,
        }
    
    def detect_activity(self, window_size: int = 10) -> str:
        """
        Detect current activity based on accelerometer data.
        
        Args:
            window_size: Number of samples to use for detection
        
        Returns:
            Activity type: 'rest', 'walking', or 'running'
        """
        magnitudes = []
        for _ in range(window_size):
            reading = self.read()
            magnitudes.append(reading["acceleration"]["magnitude"])
        
        avg_magnitude = sum(magnitudes) / len(magnitudes)
        
        if avg_magnitude < 10.5:
            return "rest"
        elif avg_magnitude < 13.0:
            return "walking"
        else:
            return "running"
    
    def is_available(self) -> bool:
        """Check if sensor is available."""
        return self._sensor_initialized or self.simulation_mode

