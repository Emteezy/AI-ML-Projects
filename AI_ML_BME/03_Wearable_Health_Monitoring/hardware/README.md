# Hardware Setup Guide

This directory contains hardware setup files and guides for the Wearable Health Monitoring System.

## Components

### Required Sensors

1. **Heart Rate & Pulse Oximetry Sensor**
   - Recommended: MAX30102
   - Alternative: MAX30100, Pulse Sensor

2. **Accelerometer/Gyroscope**
   - Recommended: MPU6050
   - Alternative: ADXL345

3. **Temperature Sensor** (Optional)
   - Recommended: DS18B20 or DHT22

### Microcontroller

- **Raspberry Pi 4** (Recommended)
- Alternative: Raspberry Pi 3B+, Arduino Uno/Nano (with modifications)

## Setup Instructions

### Raspberry Pi Setup

1. **Install OS**
   - Use Raspberry Pi OS (Raspberry Pi Imager)
   - Enable SSH and I2C in raspi-config

2. **Install Dependencies**
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-pip python3-venv
   sudo apt-get install -y i2c-tools
   ```

3. **Run Setup Script**
   ```bash
   cd hardware/raspberry_pi
   chmod +x setup.sh
   ./setup.sh
   ```

### Wiring Diagram

See `hardware/raspberry_pi/wiring.md` for detailed wiring instructions.

### Testing Sensors

```bash
python -m src.sensors.test_sensors
```

## Simulation Mode

If you don't have hardware available, the system can run in simulation mode:

```bash
python -m src.sensors.simulate --sensors heart_rate spo2 accelerometer
```

This will generate synthetic sensor data for development and testing.

## Troubleshooting

- **I2C not detected**: Run `sudo i2cdetect -y 1` to check I2C connections
- **Sensor not responding**: Check wiring and power supply
- **Permission errors**: Add user to i2c group: `sudo usermod -a -G i2c $USER`

## Additional Resources

- [Raspberry Pi GPIO Guide](https://www.raspberrypi.org/documentation/usage/gpio/)
- [I2C Protocol Guide](https://learn.sparkfun.com/tutorials/i2c/all)
- Sensor datasheets in `hardware/datasheets/`

