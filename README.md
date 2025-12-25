# Engineering Portfolio

A collection of projects I've worked on covering automation, maintenance, and AI/ML applications in biomedical engineering. Each project is fully containerized and includes working demos.

## Projects

### Automation Engineering

Industrial control systems and SCADA.

**[automation_controls_demo](Automation/automation_controls_demo/)** - PLC and SCADA system
- Soft PLC with scan-cycle logic
- HMI/SCADA dashboard
- Modbus TCP and OPC UA servers
- Plant simulator with fault injection

### Maintenance Engineering

Maintenance management and reliability.

**[maintenance_cmms_demo](Maintenance/maintenance_cmms_demo/)** - CMMS application
- Asset and PM management
- Work order tracking
- Downtime analysis and KPIs
- Root cause analysis tools

### AI/ML & Biomedical Engineering

Machine learning and healthcare applications.

**[AI_ML_BME Projects](AI_ML_BME/)**
- Medical Image Classification
- ECG Arrhythmia Detection
- Wearable Health Monitoring
- Recommendation Systems

## Quick Start

**Run all services:**
```bash
docker-compose up -d
```

**Access:**
- Automation HMI: http://localhost:8501
- CMMS Dashboard: http://localhost:8502

**Individual projects:**
```bash
make up-automation    # Automation only
make up-cmms          # CMMS only
```

**Requirements:** Docker & Docker Compose, Python 3.11+ (for local dev)

## Available Commands

```bash
make up          # Start all services
make down        # Stop all services  
make test        # Run tests
make seed        # Load sample data
make help        # Show all commands
```

## Tech Stack

**Backend:** FastAPI, SQLAlchemy, asyncio  
**Frontend:** Streamlit, Plotly  
**Protocols:** Modbus TCP, OPC UA  
**Databases:** SQLite  
**DevOps:** Docker, docker-compose  
**Testing:** pytest  

## Structure

```
├── Automation/
│   └── automation_controls_demo/
├── Maintenance/
│   └── maintenance_cmms_demo/
└── AI_ML_BME/
    ├── 01_Medical_Image_Classification/
    ├── 02_ECG_Arrhythmia_Detection/
    ├── 03_Wearable_Health_Monitoring/
    └── 04_Recommendation_System/
```

## License

MIT
