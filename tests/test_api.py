"""Integration tests for all FastAPI REST endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_api_status(client):
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OPERATIONAL"
    assert "model_stack" in data

def test_api_models_status(client):
    res = client.get("/api/models/status")
    assert res.status_code == 200
    data = res.json()
    assert "detection_tracking" in data
    assert "reid" in data
    assert "face" in data

def test_api_cameras(client):
    res = client.get("/api/cameras")
    assert res.status_code == 200
    data = res.json()
    assert "CAM-001" in data

def test_api_cross_camera_topology(client):
    res = client.get("/api/cross-camera/topology")
    assert res.status_code == 200
    data = res.json()
    assert "nodes" in data
    assert "edges" in data

def test_api_behavior_alerts(client):
    res = client.get("/api/behavior/alerts")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)

def test_api_audit_logs_and_verify(client):
    res = client.get("/api/compliance/audit-logs")
    assert res.status_code == 200
    logs = res.json()
    assert isinstance(logs, list)

    res_verify = client.get("/api/compliance/audit-verify")
    assert res_verify.status_code == 200
    v_data = res_verify.json()
    assert v_data["verified"] is True

def test_api_retention_status(client):
    res = client.get("/api/compliance/retention-status")
    assert res.status_code == 200
    data = res.json()
    assert "policy" in data
    assert "statistics" in data

def test_api_benchmark_results(client):
    res = client.get("/api/evaluation/benchmark")
    assert res.status_code == 200
    data = res.json()
    assert "cmc_rank1" in data
    assert "ablation" in data

def test_api_models_registry_and_select(client):
    # Test registry
    res = client.get("/api/models/registry")
    assert res.status_code == 200
    reg = res.json()
    assert "detection" in reg
    assert "tracking" in reg
    assert "reid" in reg
    assert "gait" in reg

    # Test dynamic selection
    sel_res = client.post("/api/models/select", json={"category": "detection", "model_id": "rtdetr"})
    assert sel_res.status_code == 200
    assert sel_res.json()["active_model"] == "rtdetr"

    # Reset back to yolo
    client.post("/api/models/select", json={"category": "detection", "model_id": "yolo"})

def test_api_models_licenses(client):
    res = client.get("/api/models/licenses")
    assert res.status_code == 200
    lic = res.json()
    assert "active_audit" in lic
    assert "full_catalog" in lic
    assert len(lic["full_catalog"]) >= 8

def test_api_cross_camera_deepstream_mtmc(client):
    res = client.get("/api/cross-camera/mtmc-benchmark")
    assert res.status_code == 200
    data = res.json()
    assert "custom_spatio_temporal_latency_ms" in data
    assert "deepstream_mtmc_latency_ms" in data
