"""
Integration tests for API
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.soft_plc.api import app


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


def test_health_check(client):
    """Test health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_get_status(client):
    """Test status endpoint"""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "mode" in data
    assert "running" in data


def test_get_io_image(client):
    """Test IO image endpoint"""
    response = client.get("/io")
    assert response.status_code == 200
    data = response.json()
    assert "inputs" in data
    assert "outputs" in data


def test_mode_change(client):
    """Test mode change endpoint"""
    response = client.post("/mode", json={"mode": "AUTO"})
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "AUTO"


def test_start_command(client):
    """Test start command"""
    # Set to AUTO mode first
    client.post("/mode", json={"mode": "AUTO"})
    
    response = client.post("/command/start")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_stop_command(client):
    """Test stop command"""
    response = client.post("/command/stop")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_get_alarms(client):
    """Test alarms endpoint"""
    response = client.get("/alarms")
    assert response.status_code == 200
    data = response.json()
    assert "alarms" in data
    assert "count" in data


def test_get_events(client):
    """Test events endpoint"""
    response = client.get("/events?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "events" in data

