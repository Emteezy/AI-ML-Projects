# Wearable Health Monitoring

IoT and edge ML system for real-time health monitoring with wearable sensors.

## Overview

Complete wearable health monitoring solution:
- IoT sensor integration (heart rate, SpO2, accelerometer)
- Edge ML inference with TensorFlow Lite
- Real-time data streaming via MQTT
- Cloud backend with FastAPI
- Monitoring dashboard with Streamlit
- Raspberry Pi/Arduino embedded systems

## Architecture

```
Wearable Device (Raspberry Pi)
├─> Sensors (HR, SpO2, Motion)
├─> Edge ML (TensorFlow Lite)
└─> MQTT streaming
    │
    ├─> Cloud Backend (FastAPI)
    │   - Data aggregation
    │   - Advanced analysis
    │   - Alert system
    │
    └─> Dashboard (Streamlit)
        - Real-time charts
        - Health metrics
        - Alerts
```

## Features

- Multi-sensor integration (MAX30102, MPU6050)
- On-device ML inference (TensorFlow Lite)
- MQTT-based real-time streaming
- Health metrics (heart rate, SpO2, activity)
- Anomaly detection algorithms
- Alert system for critical events
- Interactive dashboard with live charts
- Raspberry Pi and Arduino support

## 📁 Project Structure

```
03_Wearable_Health_Monitoring/
├── README.md                 # This file
├── PROJECT_PROPOSAL.md       # Detailed project proposal
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker container definition
├── docker-compose.yml       # Docker Compose configuration
├── .gitignore              # Git ignore rules
├── src/
│   ├── __init__.py
│   ├── sensors/
│   │   ├── __init__.py
│   │   ├── heart_rate.py    # Heart rate sensor interface
│   │   ├── pulse_oximeter.py # SpO2 sensor interface
│   │   └── accelerometer.py  # Motion sensor interface
│   ├── edge_ml/
│   │   ├── __init__.py
│   │   ├── model_converter.py # Convert models to TFLite
│   │   └── inference.py      # Edge ML inference
│   ├── iot/
│   │   ├── __init__.py
│   │   ├── mqtt_client.py    # MQTT communication
│   │   └── data_streamer.py  # Data streaming logic
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py          # FastAPI backend
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── preprocessing.py # Data preprocessing
│   │   ├── anomaly_detection.py # Anomaly detection
│   │   └── feature_extraction.py # Feature engineering
│   └── config/
│       └── settings.py      # Configuration settings
├── app/
│   └── streamlit_app.py     # Streamlit dashboard
├── hardware/
│   ├── README.md           # Hardware setup guide
│   ├── raspberry_pi/
│   │   └── setup.sh        # Raspberry Pi setup script
│   └── arduino/
│       └── sensor_reader.ino # Arduino sensor code
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_training.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_sensors.py
│   ├── test_edge_ml.py
│   └── test_api.py
├── data/                    # Sensor data (created automatically)
├── models/                  # Trained models (created automatically)
└── results/                 # Analysis results and visualizations
```

## Quick Start

**Prerequisites:** Python 3.8+, Raspberry Pi (optional), MQTT Broker

**Install:**
```bash
cd 03_Wearable_Health_Monitoring
python -m venv venv
venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

**Hardware Setup (Optional):**
- Connect sensors: MAX30102 (HR/SpO2), MPU6050 (motion)
- See `hardware/README.md` for detailed setup

**Run:**
```bash
# Start MQTT broker
mosquitto -c mosquitto.conf

# Start edge device (or simulation)
python -m src.iot.data_streamer

# Start API
python -m uvicorn src.api.main:app --reload --port 8000

# Start dashboard
streamlit run app/streamlit_app.py
```
Dashboard: http://localhost:8501

## API Endpoints

**GET /health**
```json
{"status": "healthy", "devices_connected": 2}
```

**POST /data/stream** - Receive sensor data
```json
{
  "device_id": "raspberry_pi_01",
  "heart_rate": 72,
  "spo2": 98,
  "activity": "walking"
}
```

**GET /metrics/{device_id}** - Get health metrics

**GET /alerts** - Get active alerts

## Edge ML Models

- Health status classification (Normal/Abnormal/Alert)
- Activity recognition (Resting/Walking/Running)
- Anomaly detection for unusual patterns

## Sensors Supported

- Heart Rate: MAX30102, Pulse Sensor
- Pulse Oximetry: MAX30102, MAX30100
- Motion: MPU6050, ADXL345
- Temperature: DS18B20, DHT22

## Data Pipeline

1. Read raw sensor data
2. Preprocess (filter, normalize, denoise)
3. Extract features
4. Run edge ML inference
5. Stream to cloud via MQTT
6. Cloud analysis and storage
7. Real-time dashboard updates

## Testing

```bash
pytest tests/
```

## Important Notes

- Research/educational project - not for medical diagnosis
- Consult medical professionals for health concerns
- Hardware sensors optional - runs in simulation mode
- Handle health data responsibly

## License

MIT