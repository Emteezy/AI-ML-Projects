# Quick Start Guide

This guide will help you get the Wearable Health Monitoring System up and running quickly.

## Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose (for containerized deployment)
- PostgreSQL (if running database locally)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd 03_Wearable_Health_Monitoring
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Copy the example environment file and modify as needed:

```bash
cp env.example .env
```

Edit `.env` with your configuration:
- MQTT broker settings
- Database connection string
- Device ID

### 4. (Optional) Train a model

Train a health classification model:

```bash
python src/train_model.py
```

Or use the Jupyter notebook:
```bash
jupyter notebook notebooks/model_training.ipynb
```

## Running the System

### Option 1: Docker Compose (Recommended)

Start all services with Docker Compose:

```bash
docker-compose up -d
```

This will start:
- MQTT broker (Mosquitto) on port 1883
- PostgreSQL database on port 5432
- FastAPI backend on port 8000
- Streamlit dashboard on port 8501

Access the dashboard at: http://localhost:8501

### Option 2: Run services individually

#### 1. Start MQTT Broker

Using Docker:
```bash
docker run -it -p 1883:1883 eclipse-mosquitto
```

Or install Mosquitto locally and run:
```bash
mosquitto -c mosquitto.conf
```

#### 2. Start PostgreSQL Database

Using Docker:
```bash
docker run -d \
  --name wearable_db \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=wearable_db \
  -p 5432:5432 \
  postgres:15-alpine
```

#### 3. Start FastAPI Backend

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

API documentation available at: http://localhost:8000/docs

#### 4. Start Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

Dashboard available at: http://localhost:8501

#### 5. Run Edge Device (Simulation)

```bash
python run_edge_device.py
```

Or:
```bash
python src/edge_device.py
```

This will:
- Read from sensors (simulated)
- Run ML inference (if model is available)
- Detect anomalies
- Stream data to MQTT broker

## Using the System

### 1. View Real-time Data

Open the Streamlit dashboard at http://localhost:8501 to view:
- Real-time sensor readings
- Health status predictions
- Anomaly alerts
- Data visualizations

### 2. Access API Endpoints

The FastAPI backend provides REST endpoints:
- `GET /api/readings` - Get sensor readings
- `GET /api/health-status` - Get health status
- `GET /api/alerts` - Get anomaly alerts
- `GET /api/statistics` - Get system statistics

Full API documentation: http://localhost:8000/docs

### 3. Monitor MQTT Messages

You can subscribe to MQTT topics to monitor data:

```bash
mosquitto_sub -h localhost -t "wearable/health/#" -v
```

Topics:
- `wearable/health/{device_id}/sensors/{sensor_type}` - Sensor readings
- `wearable/health/{device_id}/health/status` - Health status
- `wearable/health/{device_id}/alerts/anomaly` - Anomaly alerts

## Simulation Mode

By default, the system runs in simulation mode (no physical hardware required). Sensor data is generated synthetically.

To disable simulation mode and use real hardware:
1. Set `SIMULATION_MODE=false` in `.env`
2. Connect physical sensors (MAX30102, MPU6050)
3. Ensure proper I2C connections on Raspberry Pi

## Troubleshooting

### MQTT Connection Issues

- Ensure MQTT broker is running
- Check MQTT broker address and port in configuration
- Verify firewall settings allow MQTT traffic

### Database Connection Issues

- Ensure PostgreSQL is running
- Check database connection string in `.env`
- Verify database credentials

### Model Not Found

- Train a model first using `python src/train_model.py`
- Ensure model file exists in `models/` directory
- Check model path in configuration

## Next Steps

- Train models with your own data
- Connect physical sensors for real data collection
- Customize health thresholds in `src/config/settings.py`
- Extend the dashboard with additional visualizations
- Deploy to production environment

For more information, see the main [README.md](README.md).

