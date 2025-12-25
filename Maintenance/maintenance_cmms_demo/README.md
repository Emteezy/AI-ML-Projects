# Maintenance CMMS

A comprehensive Computerized Maintenance Management System for tracking assets, work orders, preventive maintenance, and reliability metrics.

## Overview

This CMMS application provides end-to-end maintenance management capabilities including asset tracking, PM scheduling, work order management, downtime analysis, and KPI reporting. Built for maintenance and reliability engineers who need practical tools for equipment management.

## Features

- **Asset Management** - Track equipment with criticality levels and specifications
- **PM Scheduling** - Frequency-based preventive maintenance planning
- **Work Orders** - Full lifecycle management for PM and corrective maintenance
- **Downtime Tracking** - Record and analyze equipment failures
- **Spare Parts** - Inventory management with min/max levels
- **Root Cause Analysis** - 5-Why and Fishbone analysis tools
- **KPI Dashboard** - MTTR, MTBF, availability, and PM compliance metrics
- **Reporting** - Export data and generate maintenance reports

## Quick Start

### Using Docker

```bash
docker-compose up -d

# Load sample data
docker-compose exec cmms_api python seed_data.py
```

Access the dashboard at http://localhost:8502

### Local Development

```bash
pip install -r requirements.txt

# Terminal 1: Start API
python -m uvicorn src.api:app --port 8003

# Terminal 2: Start UI
streamlit run app/cmms_dashboard.py

# Load sample data
python seed_data.py
```

## Sample Data

The seed script creates:
- 5 assets (conveyor, pump, motor, compressor, robot)
- 3 PM plans with different frequencies
- 10 work orders (mix of PM and CM)
- 5 downtime incidents
- Spare parts inventory
- Sample root cause analysis

## Dashboard Pages

1. **Dashboard** - KPIs and system overview
2. **Assets** - Equipment inventory and details
3. **PM Scheduler** - View and create PM plans
4. **Work Orders** - Create, assign, and close work orders
5. **Downtime** - Track failure incidents
6. **Spares** - Manage parts inventory
7. **RCA** - Conduct root cause analysis
8. **Reports** - Export and visualize data

## KPIs Calculated

- **MTTR** (Mean Time To Repair) - Average repair time
- **MTBF** (Mean Time Between Failures) - Reliability metric
- **Downtime Hours** - Total equipment downtime
- **PM Compliance** - On-time PM completion rate
- **Failure Analysis** - Top failure modes and trends

## API

RESTful API with full CRUD operations:

- Assets: `/assets`
- Work Orders: `/work-orders`
- PM Plans: `/pm-plans`
- Downtime: `/downtime`
- Spare Parts: `/spare-parts`
- KPIs: `/kpis`

Interactive docs: http://localhost:8003/docs

## Testing

```bash
pytest tests/ -v
```

## Database

SQLite database with async support. Schema includes:
- Assets
- PM Plans
- Work Orders
- Downtime Incidents
- Spare Parts
- Root Cause Analysis

## Project Structure

```
maintenance_cmms_demo/
├── src/
│   ├── models.py          # Database models
│   ├── api.py            # FastAPI backend
│   ├── kpi_calculator.py # Reliability metrics
│   └── database.py       # DB setup
├── app/
│   └── cmms_dashboard.py # Streamlit UI
├── tests/                # Test suite
└── seed_data.py         # Sample data generator
```

## License

MIT
