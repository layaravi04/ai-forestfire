import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "HEALTHY"

def test_models_list():
    response = client.get("/models")
    assert response.status_code == 200
    assert len(response.json()) == 8

def test_model_comparison():
    response = client.get("/model-comparison")
    assert response.status_code == 200
    assert len(response.json()) == 8

def test_satellite_hotspots():
    response = client.get("/satellite/hotspots")
    assert response.status_code == 200
    assert "hotspots" in response.json()
