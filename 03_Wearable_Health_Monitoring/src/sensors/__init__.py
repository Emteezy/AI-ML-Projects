"""Sensor interfaces for wearable health monitoring."""

from .heart_rate import HeartRateSensor
from .pulse_oximeter import PulseOximeterSensor
from .accelerometer import AccelerometerSensor

__all__ = [
    "HeartRateSensor",
    "PulseOximeterSensor",
    "AccelerometerSensor",
]
