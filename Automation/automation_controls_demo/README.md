# Automation Controls Demo

A full-featured industrial PLC and SCADA system simulation demonstrating real-world automation engineering concepts and practices.

## Overview

This project simulates an industrial conveyor cell with a soft PLC controller, plant simulator, HMI/SCADA interface, and industrial communication protocols. It's designed to showcase practical automation engineering skills including PLC programming, SCADA development, and industrial networking.

## Features

- **Soft PLC Controller** - Scan-based PLC logic with 100ms cycle time
- **Multi-Mode Operation** - AUTO, MANUAL, and FAULT modes with proper interlocks
- **Alarm Management** - Multi-level alarm system with acknowledge/clear workflow
- **HMI/SCADA Dashboard** - Real-time monitoring and control interface
- **Industrial Protocols** - Modbus TCP and OPC UA servers
- **Data Historian** - Time-series data logging and trending
- **Plant Simulator** - Physics-based conveyor simulation with fault injection
- **Documentation** - Auto-generated commissioning documents

## Quick Start

### Using Docker

```bash
docker-compose up -d
```

Access the HMI at http://localhost:8501

### Local Development

```bash
pip install -r requirements.txt

# Terminal 1: Start PLC
python -m uvicorn src.soft_plc.api:app --port 8001

# Terminal 2: Start Plant Simulator  
python -m uvicorn src.plant_sim.simulator:app --port 8002

# Terminal 3: Start HMI
streamlit run app/hmi_dashboard.py
```

## Demo Script

Run the automated demo scenario:

```bash
python demo.py
```

This demonstrates:
- System startup and normal operation
- Fault detection (conveyor jam)
- Alarm handling by operator
- System recovery procedures

## Architecture

The system consists of three main components:

1. **Soft PLC** (port 8001) - Core control logic with scan cycle execution
2. **Plant Simulator** (port 8002) - Simulates physical equipment and sensors
3. **HMI Dashboard** (port 8501) - Operator interface and monitoring

### Communication

- **Modbus TCP** on port 5020 - Standard industrial protocol
- **OPC UA** on port 4840 - Modern industrial IoT protocol
- **REST API** - For HMI and external integrations

## Project Structure

```
automation_controls_demo/
├── src/
│   ├── soft_plc/          # PLC core logic and API
│   ├── comms/             # Modbus and OPC UA gateways
│   └── plant_sim/         # Plant physics simulation
├── app/                   # HMI dashboard
├── tests/                 # Unit and integration tests
├── docs/                  # Generated documentation
└── demo.py               # Automated demo script
```

## Testing

```bash
pytest tests/ -v
```

## Documentation

Generated commissioning documentation includes:
- I/O lists
- Alarm lists
- Cause & Effect matrices
- FAT/SAT checklists
- Control philosophy
- Protocol register maps

Generate docs: `python src/docs_generator.py`

## API Documentation

Interactive API docs available at:
- PLC API: http://localhost:8001/docs
- Plant Sim: http://localhost:8002/docs

## License

MIT
