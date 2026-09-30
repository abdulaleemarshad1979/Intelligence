"""Comprehensive test suite for 600-camera fleet scaling and ICSee camera integration."""

import os
import pytest
from fastapi.testclient import TestClient
from app.main import app, CCTV_CAMERAS_REGISTRY, load_camera_config
from app.ingestion.stream_manager import build_icsee_rtsp_url, get_stream_manager
from app.discovery.topology_generator import generate_600_camera_topology, export_600_camera_yaml
from app.reid.cross_camera import CrossCameraTracker

@pytest.fixture
def client():
    return TestClient(app)


def test_600_camera_topology_generation():
    """Verify that 600 unique, calibrated cameras are generated with realistic parameters."""
    cams = generate_600_camera_topology(target_count=600)
    assert len(cams) >= 600
    assert "CAM-001" in cams
    assert "CAM-600" in cams

    # Check CAM-001 preserved reference calibration
    cam1 = cams["CAM-001"]
    assert "District Hospital" in cam1["name"] or "Hospital" in cam1["location"]
    assert len(cam1["resolution"]) == 2

    # Check CAM-600 valid structure
    cam600 = cams["CAM-600"]
    assert cam600["mounting_height_m"] >= 3.5
    assert cam600["tilt_angle_deg"] >= 20.0
    assert cam600["latitude"] > 16.0
    assert cam600["longitude"] > 81.0
    assert len(cam600["adjacent_cameras"]) > 0


def test_cctv_registry_scale_in_api(client):
    """Verify that CCTV_CAMERAS_REGISTRY has 600 cameras and API endpoints return them."""
    assert len(CCTV_CAMERAS_REGISTRY) >= 600
    assert CCTV_CAMERAS_REGISTRY[0]["camera_id"] == "CAM-001"
    assert CCTV_CAMERAS_REGISTRY[599]["camera_id"] == "CAM-600"

    # 1. Test /api/cctv/cameras
    res = client.get("/api/cctv/cameras")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["total"] >= 600
    assert len(data["cameras"]) >= 600

    # Test filtering by sector
    res_hosp = client.get("/api/cctv/cameras?sector=Hospital")
    assert res_hosp.status_code == 200
    data_hosp = res_hosp.json()
    assert data_hosp["total"] > 0
    for c in data_hosp["cameras"]:
        assert c["sector"] == "Hospital"

    # Test pagination
    res_page = client.get("/api/cctv/cameras?limit=24&offset=0")
    assert res_page.status_code == 200
    assert len(res_page.json()["cameras"]) == 24


def test_cameras_dashboard_endpoint_600(client):
    """Verify that /cameras returns 600 stream configs for the Command Center dashboard."""
    res = client.get("/cameras")
    assert res.status_code == 200
    cams = res.json()
    assert len(cams) >= 600
    assert cams[0]["id"] == "CAM-001"
    assert cams[599]["id"] == "CAM-600"
    assert cams[0]["status"] == "online"


def test_icsee_rtsp_url_builder():
    """Verify Xiongmai / ICSee RTSP stream URL generation with credentials."""
    url = build_icsee_rtsp_url(
        ip="192.168.1.111",
        port=554,
        username="hmpw",
        password="r7h3m2",
        stream_type="stream0"
    )
    assert url == "rtsp://hmpw:r7h3m2@192.168.1.111:554/stream0"

    # Sub-stream profile
    url_sub = build_icsee_rtsp_url(
        ip="192.168.1.111",
        port=554,
        username="hmpw",
        password="r7h3m2",
        stream_type="stream1"
    )
    assert url_sub == "rtsp://hmpw:r7h3m2@192.168.1.111:554/stream1"


def test_api_connect_icsee_camera(client):
    """Test POST /api/cameras/connect_icsee endpoint."""
    payload = {
        "camera_id": "CAM-001",
        "ip": "192.168.1.111",
        "port": 554,
        "username": "hmpw",
        "password": "r7h3m2",
        "stream_type": "stream0",
        "name": "Front Gate ICSee Cam"
    }
    res = client.post("/api/cameras/connect_icsee", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["camera_id"] == "CAM-001"
    assert "rtsp://" in data["stream_url"]
    assert "192.168.1.111" in data["stream_url"]


def test_cross_camera_tracker_600_topology():
    """Verify that CrossCameraTracker initializes and computes travel windows across 600 nodes."""
    configs = load_camera_config()
    assert len(configs) >= 600
    tracker = CrossCameraTracker(configs)
    assert len(tracker.cameras) >= 600

    # Test travel window between CAM-001 and CAM-002
    dist_m, min_s, max_s = tracker.compute_travel_window("CAM-001", "CAM-002")
    assert dist_m > 0
    assert min_s > 0
    assert max_s > min_s

    # Test travel window across distant cameras
    dist_far, min_far, max_far = tracker.compute_travel_window("CAM-001", "CAM-500")
    assert dist_far > 0
