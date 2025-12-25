# Engineering Portfolio

A collection of projects spanning automation, maintenance engineering, and AI/ML applications in biomedical engineering. These projects demonstrate end-to-end development from concept to deployment, with a focus on practical, production-ready solutions.

## 🎯 Overview

This repository showcases my work across three engineering domains:

- **Industrial Automation** - Control systems, SCADA, and industrial protocols
- **Maintenance Engineering** - CMMS applications and reliability analysis
- **AI/ML in Healthcare** - Deep learning models for medical applications

Each project is fully containerized with Docker and includes complete documentation, tests, and working demos.

---

## 📂 Projects

### 🏭 Automation Engineering

#### [Automation Controls Demo](Automation/automation_controls_demo/)

A complete soft PLC and SCADA system demonstrating industrial automation concepts.

**Key Features:**
- Soft PLC with configurable scan-cycle logic
- Real-time HMI/SCADA dashboard built with Streamlit
- Modbus TCP and OPC UA server implementations
- Plant simulator with fault injection capabilities
- Historian for time-series data logging

**Tech Stack:** Python, asyncio, pymodbus, opcua, Streamlit

---

### 🔧 Maintenance Engineering

#### [CMMS Application](Maintenance/maintenance_cmms_demo/)

Computerized Maintenance Management System for asset tracking and reliability analysis.

**Key Features:**
- Asset and preventive maintenance management
- Work order lifecycle tracking
- Real-time KPI dashboards (MTBF, MTTR, OEE)
- Root cause analysis tools
- Downtime tracking and reporting

**Tech Stack:** FastAPI, SQLAlchemy, Streamlit, Plotly

---

### 🏥 AI/ML & Biomedical Engineering

#### [Medical Image Classification](AI_ML_BME/01_Medical_Image_Classification/)

Deep learning model for classifying chest X-rays using transfer learning with ResNet architectures.

**Features:**
- ResNet-based transfer learning (ResNet18/34/50)
- GradCAM visualization for model explainability
- REST API for inference
- Web interface for demo

#### [ECG Arrhythmia Detection](AI_ML_BME/02_ECG_Arrhythmia_Detection/)

Real-time ECG signal processing and arrhythmia classification using deep learning.

**Features:**
- Signal preprocessing and filtering
- LSTM and Transformer model architectures
- Multi-class arrhythmia detection
- Real-time inference API

#### [Wearable Health Monitoring](AI_ML_BME/03_Wearable_Health_Monitoring/)

IoT-based health monitoring system with edge ML capabilities.

**Features:**
- Sensor data simulation (heart rate, SpO2, accelerometer)
- MQTT-based real-time streaming
- Edge ML with TFLite model conversion
- Anomaly detection algorithms
- Web dashboard for monitoring

#### [Recommendation System](AI_ML_BME/04_Recommendation_System/)

Hybrid recommendation system combining multiple approaches.

**Features:**
- Collaborative filtering
- Content-based filtering
- Matrix factorization (SVD)
- Neural collaborative filtering
- Hybrid ensemble approach

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for local development)
- 8GB RAM recommended

### Running All Services

```bash
# Clone the repository
git clone https://github.com/Emteezy/AI-ML-Projects.git
cd AI-ML-Projects

# Start all services
docker-compose up -d

# Access the dashboards
# Automation HMI: http://localhost:8501
# CMMS Dashboard: http://localhost:8502
```

### Running Individual Projects

```bash
# Automation only
make up-automation

# CMMS only
make up-cmms

# Stop all services
make down
```

### Available Make Commands

```bash
make up          # Start all services
make down        # Stop all services
make test        # Run all tests
make seed        # Load sample data
make clean       # Clean up generated files
make help        # Show all commands
```

---

## 🛠️ Tech Stack

**Languages:** Python 3.11+

**Backend:**
- FastAPI - Modern web framework
- SQLAlchemy - ORM and database toolkit
- asyncio - Asynchronous programming

**Frontend:**
- Streamlit - Interactive dashboards
- Plotly - Data visualization

**ML/DL:**
- PyTorch - Deep learning framework
- scikit-learn - Machine learning toolkit
- TensorFlow Lite - Edge deployment

**Industrial:**
- pymodbus - Modbus TCP/RTU
- opcua - OPC UA server/client

**DevOps:**
- Docker & docker-compose
- pytest - Testing framework
- GitHub Actions - CI/CD (planned)

**IoT:**
- MQTT - Message broker protocol
- SQLite - Embedded database

---

## 📁 Repository Structure

```
AI_ML_PORTFOLIO/
│
├── Automation/              # Industrial automation projects
│   └── automation_controls_demo/
│       ├── app/            # HMI dashboard
│       ├── src/            # Core logic (PLC, comms, simulator)
│       └── tests/          # Unit tests
│
├── Maintenance/            # Maintenance engineering projects
│   └── maintenance_cmms_demo/
│       ├── app/            # CMMS dashboard
│       ├── src/            # Backend (API, models, KPIs)
│       └── tests/          # Unit tests
│
└── AI_ML_BME/             # AI/ML biomedical projects
    ├── 01_Medical_Image_Classification/
    ├── 02_ECG_Arrhythmia_Detection/
    ├── 03_Wearable_Health_Monitoring/
    └── 04_Recommendation_System/
```

Each project includes:
- `README.md` - Project documentation
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container configuration
- `docker-compose.yml` - Service orchestration
- `tests/` - Test suite

---

## 🧪 Testing

Run tests for all projects:

```bash
make test
```

Run tests for specific project:

```bash
cd Automation/automation_controls_demo
pytest tests/

cd Maintenance/maintenance_cmms_demo
pytest tests/
```

---

## 📝 Project Notes

### Development Approach

These projects follow software engineering best practices:
- Modular, maintainable code structure
- Comprehensive documentation
- Unit and integration tests
- Containerization for reproducibility
- REST APIs for integration

### Future Enhancements

- [ ] Add CI/CD pipeline with GitHub Actions
- [ ] Kubernetes deployment configurations
- [ ] Real hardware integration for IoT project
- [ ] Model performance monitoring
- [ ] Enhanced documentation with tutorials

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🤝 Contributing

While this is a personal portfolio, suggestions and feedback are welcome! Feel free to:
- Open an issue for bugs or suggestions
- Submit pull requests for improvements
- Star the repo if you find it useful

---

## 📧 Contact

For questions or collaboration opportunities, feel free to reach out via GitHub issues or connect with me on [LinkedIn](https://linkedin.com).

---

**Note:** The AI/ML models included are for demonstration purposes. For production medical applications, proper validation, regulatory approval, and clinical testing would be required.
