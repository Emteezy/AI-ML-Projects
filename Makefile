.PHONY: help up down restart logs clean test seed docs

help:
	@echo "Industrial Portfolio - Makefile Commands"
	@echo ""
	@echo "General Commands:"
	@echo "  make up              - Start all services"
	@echo "  make down            - Stop all services"
	@echo "  make restart         - Restart all services"
	@echo "  make logs            - View logs from all services"
	@echo "  make clean           - Clean up containers and volumes"
	@echo ""
	@echo "Development:"
	@echo "  make test            - Run tests for all projects"
	@echo "  make seed            - Seed databases with sample data"
	@echo "  make docs            - Generate commissioning documentation"
	@echo ""
	@echo "Individual Projects:"
	@echo "  make up-automation   - Start automation controls demo"
	@echo "  make up-cmms         - Start maintenance CMMS demo"
	@echo ""
	@echo "Access Points:"
	@echo "  Automation HMI:      http://localhost:8501"
	@echo "  Automation PLC API:  http://localhost:8001"
	@echo "  CMMS UI:             http://localhost:8502"
	@echo "  CMMS API:            http://localhost:8003"

# General commands
up:
	docker-compose up -d

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	rm -rf Automation/automation_controls_demo/data/*.db
	rm -rf Maintenance/maintenance_cmms_demo/data/*.db

# Testing
test:
	@echo "Running automation controls tests..."
	cd Automation/automation_controls_demo && pytest tests/ -v
	@echo "Running CMMS tests..."
	cd Maintenance/maintenance_cmms_demo && pytest tests/ -v

test-automation:
	cd Automation/automation_controls_demo && pytest tests/ -v

test-cmms:
	cd Maintenance/maintenance_cmms_demo && pytest tests/ -v

# Seeding data
seed: seed-cmms

seed-cmms:
	docker-compose exec cmms_api python seed_data.py

# Documentation
docs:
	cd Automation/automation_controls_demo && python src/docs_generator.py

# Individual project commands
up-automation:
	docker-compose up -d automation_plc automation_plant_sim automation_hmi

up-cmms:
	docker-compose up -d cmms_api cmms_ui

down-automation:
	docker-compose stop automation_plc automation_plant_sim automation_hmi

down-cmms:
	docker-compose stop cmms_api cmms_ui

# Demo scripts
demo-automation:
	@echo "Make sure services are running first (make up-automation)"
	@echo "Waiting for services to be ready..."
	@sleep 5
	cd Automation/automation_controls_demo && python demo.py

demo-cmms:
	docker-compose exec cmms_api python seed_data.py

# Build commands
build:
	docker-compose build

build-automation:
	docker-compose build automation_plc automation_plant_sim automation_hmi

build-cmms:
	docker-compose build cmms_api cmms_ui

# Development mode (run locally without Docker)
dev-automation-plc:
	cd Automation/automation_controls_demo && python -m uvicorn src.soft_plc.api:app --reload --port 8001

dev-automation-sim:
	cd Automation/automation_controls_demo && python -m uvicorn src.plant_sim.simulator:app --reload --port 8002

dev-automation-hmi:
	cd Automation/automation_controls_demo && streamlit run app/hmi_dashboard.py --server.port 8501

dev-cmms-api:
	cd Maintenance/maintenance_cmms_demo && python -m uvicorn src.api:app --reload --port 8003

dev-cmms-ui:
	cd Maintenance/maintenance_cmms_demo && streamlit run app/cmms_dashboard.py --server.port 8502
