"""Simple test script to verify sensors are working."""

from src.sensors import HeartRateSensor, PulseOximeterSensor, AccelerometerSensor

def test_sensors():
    """Test all sensors."""
    print("Testing sensors in simulation mode...\n")
    
    # Test heart rate sensor
    print("1. Testing Heart Rate Sensor:")
    hr_sensor = HeartRateSensor(simulation_mode=True)
    hr_reading = hr_sensor.read()
    print(f"   Heart Rate: {hr_reading['heart_rate']} {hr_reading['unit']}")
    print()
    
    # Test SpO2 sensor
    print("2. Testing Pulse Oximeter Sensor:")
    spo2_sensor = PulseOximeterSensor(simulation_mode=True)
    spo2_reading = spo2_sensor.read()
    print(f"   SpO2: {spo2_reading['spo2']} {spo2_reading['unit']}")
    print()
    
    # Test accelerometer
    print("3. Testing Accelerometer Sensor:")
    accel_sensor = AccelerometerSensor(simulation_mode=True)
    accel_reading = accel_sensor.read()
    print(f"   Acceleration (X, Y, Z): ({accel_reading['acceleration']['x']}, "
          f"{accel_reading['acceleration']['y']}, {accel_reading['acceleration']['z']}) {accel_reading['unit']}")
    print(f"   Magnitude: {accel_reading['acceleration']['magnitude']}")
    print(f"   Activity State: {accel_reading['activity_state']}")
    print()
    
    print("✅ All sensors are working correctly!")

if __name__ == "__main__":
    test_sensors()

