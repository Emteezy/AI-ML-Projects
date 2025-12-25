# Wearable Health Monitoring System 🏥💻

A **full-stack IoT + ML system** for real-time health monitoring using wearable sensors. This project demonstrates edge ML, IoT integration, real-time data processing, and embedded systems design.

## 🎯 Overview

This project provides a complete solution for wearable health monitoring, featuring:
- **IoT Sensor Integration** (Heart Rate, Pulse Oximetry, Accelerometer)
- **Edge Machine Learning** (TensorFlow Lite for on-device inference)
- **Real-time Data Streaming** (MQTT protocol)
- **Cloud Backend** (FastAPI for data aggregation and analysis)
- **Monitoring Dashboard** (Streamlit for visualization)
- **Embedded System Design** (Raspberry Pi/Arduino integration)

## 🏗️ Architecture

```
┌─────────────────┐
│  Wearable Device│
│  (Raspberry Pi) │
│  ┌───────────┐  │
│  │ Sensors:  │  │
│  │ - HR      │  │
│  │ - SpO2    │  │
│  │ - Motion  │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ Edge ML   │  │
│  │ (TFLite)  │  │
│  └─────┬─────┘  │
└────────┼────────┘
         │
         │ MQTT
         │
┌────────▼─────────────────┐
│  Cloud Backend (FastAPI) │
│  - Data Aggregation      │
│  - Advanced Analysis     │
│  - Alert System          │
└────────┬─────────────────┘
         │
         │ HTTP/REST
         │
┌────────▼─────────────┐
│  Dashboard (Streamlit)│
│  - Real-time Charts  │
│  - Health Metrics    │
│  - Alerts/Notifications│
└──────────────────────┘
```

## ✨ Key Features

- ✅ **IoT Sensor Integration**: Connect and read from multiple health sensors
- ✅ **Edge ML Inference**: On-device predictions using TensorFlow Lite
- ✅ **Real-time Streaming**: MQTT-based data transmission
- ✅ **Health Metrics**: Heart rate, SpO2, activity tracking
- ✅ **Anomaly Detection**: Detect abnormal health patterns
- ✅ **Alert System**: Notifications for critical health events
- ✅ **Data Visualization**: Interactive dashboard with real-time charts
- ✅ **Embedded Systems**: Raspberry Pi/Arduino integration

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

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Raspberry Pi (optional, for hardware deployment)
- Sensors: Heart Rate, Pulse Oximeter, Accelerometer (optional)
- MQTT Broker (e.g., Mosquitto)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd 03_Wearable_Health_Monitoring
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # Windows PowerShell
   venv\Scripts\Activate.ps1
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Hardware Setup (Optional)

For full IoT functionality, set up hardware sensors:

1. **Raspberry Pi Setup**
   ```bash
   cd hardware/raspberry_pi
   ./setup.sh
   ```

2. **Connect Sensors**
   - Heart Rate Sensor (e.g., MAX30102)
   - Pulse Oximeter
   - Accelerometer/Gyroscope (e.g., MPU6050)

3. **See** `hardware/README.md` for detailed hardware setup instructions

### Running the System

1. **Start MQTT Broker** (if using local broker)
   ```bash
   mosquitto -c mosquitto.conf
   ```

2. **Run Edge Device** (Raspberry Pi or simulation)
   ```bash
   python -m src.iot.data_streamer
   ```

3. **Start Backend API**
   ```bash
   python -m uvicorn src.api.main:app --reload --port 8000
   ```

4. **Launch Dashboard**
   ```bash
   streamlit run app/streamlit_app.py
   ```

The dashboard will be available at: http://localhost:8501

## 📖 API Documentation

### Endpoints

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "devices_connected": 2,
  "active_sessions": 5
}
```

#### `POST /data/stream`
Receive sensor data from edge devices.

**Request:**
```json
{
  "device_id": "raspberry_pi_01",
  "timestamp": "2024-01-15T10:30:00Z",
  "heart_rate": 72,
  "spo2": 98,
  "temperature": 36.5,
  "activity": "walking"
}
```

#### `GET /metrics/{device_id}`
Get health metrics for a specific device.

#### `GET /alerts`
Get active health alerts.

## 🎓 Technical Details

### Edge ML Models

- **Health Status Classification**: Normal/Abnormal/Aler
- **Activity Recognition**: Resting/Walking/Running
- **Anomaly Detection**: Detect unusual patterns

### Sensors Supported

- **Heart Rate**: MAX30102, Pulse Sensor
- **Pulse Oximetry**: MAX30102, MAX30100
- **Motion**: MPU6050, ADXL345
- **Temperature**: DS18B20, DHT22

### Data Processing Pipeline

1. **Sensor Reading**: Read raw sensor data
2. **Preprocessing**: Filter, normalize, remove noise
3. **Feature Extraction**: Extract relevant features
4. **Edge Inference**: Run ML models on device
5. **Data Streaming**: Send to cloud via MQTT
6. **Cloud Analysis**: Advanced analysis and storage
7. **Visualization**: Real-time dashboard updates

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
black src/ tests/ app/
flake8 src/ tests/ app/
```

## 📊 Use Cases

- **Remote Patient Monitoring**: Monitor patients at home
- **Fitness Tracking**: Track workout and recovery metrics
- **Elderly Care**: Monitor elderly family members
- **Clinical Research**: Collect health data for studies
- **Personal Health**: Track personal health metrics

## ⚠️ Important Notes

- **This is a research/educational project** - Not for medical diagnosis
- **Always consult medical professionals** for health concerns
- **Hardware sensors are optional** - System can run in simulation mode
- **Ensure data privacy** - Handle health data responsibly

## 🚧 Future Enhancements

- [ ] Support for additional sensors (ECG, EMG)
- [ ] Mobile app for iOS/Android
- [ ] Cloud deployment (AWS IoT, Google Cloud IoT)
- [ ] Machine learning model training pipeline
- [ ] Multi-user support with authentication
- [ ] Integration with fitness trackers (Fitbit, Apple Watch)
- [ ] Advanced analytics and trend analysis

## 📄 License

This project is open source and available under the MIT License.

---

**Project Status**: 🚧 In Development  
**Last Updated**: 2024-12-21

