"""Tests for sensor interfaces."""

import pytest
from src.sensors import HeartRateSensor, PulseOximeterSensor, AccelerometerSensor


def test_heart_rate_sensor():
    """Test heart rate sensor."""
    sensor = HeartRateSensor(simulation_mode=True)
    assert sensor.is_available()
    
    reading = sensor.read()
    assert "heart_rate" in reading
    assert "timestamp" in reading
    assert "unit" in reading
    assert reading["unit"] == "bpm"
    assert 40 <= reading["heart_rate"] <= 200


def test_pulse_oximeter_sensor():
    """Test pulse oximeter sensor."""
    sensor = PulseOximeterSensor(simulation_mode=True)
    assert sensor.is_available()
    
    reading = sensor.read()
    assert "spo2" in reading
    assert "timestamp" in reading
    assert "unit" in reading
    assert reading["unit"] == "%"
    assert 70 <= reading["spo2"] <= 100


def test_accelerometer_sensor():
    """Test accelerometer sensor."""
    sensor = AccelerometerSensor(simulation_mode=True)
    assert sensor.is_available()
    
    reading = sensor.read()
    assert "acceleration" in reading
    assert "gyroscope" in reading
    assert "timestamp" in reading
    assert "x" in reading["acceleration"]
    assert "y" in reading["acceleration"]
    assert "z" in reading["acceleration"]
    assert "magnitude" in reading["acceleration"]

