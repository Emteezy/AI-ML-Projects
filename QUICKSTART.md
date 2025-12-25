# Quick Start

## Prerequisites

- Docker Desktop running
- Ports 8001-8003, 8501-8502 available

## Start Everything

```bash
docker-compose up -d
```

Wait 30 seconds for services to initialize.

## Load Sample Data

```bash
docker-compose exec cmms_api python seed_data.py
```

## Access Applications

- **Automation HMI**: http://localhost:8501
- **CMMS Dashboard**: http://localhost:8502

## Try the Automation Demo

1. Open http://localhost:8501
2. Switch to AUTO mode
3. Click START button
4. Go to Diagnostics → Inject JAM fault
5. Watch alarm system respond
6. Clear fault and reset

Or run automated demo:
```bash
docker-compose exec automation_plc python demo.py
```

## Explore CMMS

1. Open http://localhost:8502
2. Check Dashboard for KPIs
3. Browse Assets, Work Orders, PM Plans
4. View Downtime and RCA sections

## Stop Services

```bash
docker-compose down
```

## Local Development

### Automation

```bash
cd Automation/automation_controls_demo
pip install -r requirements.txt

# Three terminals:
python -m uvicorn src.soft_plc.api:app --port 8001
python -m uvicorn src.plant_sim.simulator:app --port 8002
streamlit run app/hmi_dashboard.py
```

### CMMS

```bash
cd Maintenance/maintenance_cmms_demo
pip install -r requirements.txt

# Two terminals:
python -m uvicorn src.api:app --port 8003
streamlit run app/cmms_dashboard.py
```

## Troubleshooting

**Services won't start:**
```bash
docker-compose down
docker-compose up --build -d
```

**Port conflicts:**
```bash
netstat -ano | findstr :8501
```

**Database issues:**
```bash
docker-compose down -v
docker-compose up -d
docker-compose exec cmms_api python seed_data.py
```
