"""Tests for CMMS API"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.api import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["service"] == "CMMS API"


def test_health_check(client):
    """Test health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_create_asset(client):
    """Test asset creation"""
    asset_data = {
        "asset_tag": "TEST-001",
        "name": "Test Asset",
        "category": "Pump",
        "criticality": "HIGH"
    }
    response = client.post("/assets", json=asset_data)
    assert response.status_code == 200
    data = response.json()
    assert data["asset_tag"] == "TEST-001"
    assert data["name"] == "Test Asset"


def test_list_assets(client):
    """Test listing assets"""
    response = client.get("/assets")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_create_work_order(client):
    """Test work order creation"""
    # First create an asset
    asset_data = {
        "asset_tag": "WO-TEST-001",
        "name": "WO Test Asset",
        "criticality": "MEDIUM"
    }
    asset_response = client.post("/assets", json=asset_data)
    asset_id = asset_response.json()["id"]
    
    # Create work order
    wo_data = {
        "asset_id": asset_id,
        "wo_type": "PM",
        "priority": "HIGH",
        "title": "Test Work Order",
        "description": "Test description"
    }
    response = client.post("/work-orders", json=wo_data)
    assert response.status_code == 200
    data = response.json()
    assert "wo_number" in data
    assert data["title"] == "Test Work Order"


def test_list_work_orders(client):
    """Test listing work orders"""
    response = client.get("/work-orders")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_filter_work_orders_by_status(client):
    """Test filtering work orders by status"""
    response = client.get("/work-orders?status=OPEN")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_create_spare_part(client):
    """Test spare part creation"""
    part_data = {
        "part_number": "TEST-PART-001",
        "description": "Test Part",
        "quantity_on_hand": 10,
        "min_quantity": 5,
        "unit_cost": 25.50
    }
    response = client.post("/spare-parts", json=part_data)
    assert response.status_code == 200
    data = response.json()
    assert data["part_number"] == "TEST-PART-001"


def test_get_kpis(client):
    """Test KPI endpoint"""
    response = client.get("/kpis?days=30")
    assert response.status_code == 200
    data = response.json()
    assert "mttr_hours" in data
    assert "mtbf_hours" in data
    assert "total_downtime_hours" in data
    assert "pm_compliance_percent" in data

