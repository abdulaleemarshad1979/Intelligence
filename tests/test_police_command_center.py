"""Test Suite for Police Command Center & Clean Surveillance Architecture."""

import pytest
import numpy as np
import cv2
import json
from fastapi.testclient import TestClient

from app.main import app, STREAM_OVERLAY_CONFIG
from app.database.models import CriminalRecord, TrackObservation
from app.database.repository import Repository
from app.fusion.evidence import EvidenceFusionEngine


@pytest.fixture
def client():
    return TestClient(app)


def test_clean_video_stream_overlay_config(client):
    """Test overlay config endpoint and verify default clean mode."""
    res = client.get("/api/stream/overlay")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["overlay"]["show_skeleton"] is False
    assert data["overlay"]["show_face_mesh"] is False

    # Update overlay mode to minimal
    res2 = client.post("/api/stream/overlay", json={"mode": "minimal"})
    assert res2.status_code == 200
    assert res2.json()["overlay"]["mode"] == "minimal"

    # Reset back to clean
    client.post("/api/stream/overlay", json={"mode": "clean"})
    assert STREAM_OVERLAY_CONFIG["mode"] == "clean"


def test_cctv_cameras_matrix(client):
    """Test 16-camera district CCTV matrix endpoint."""
    res = client.get("/api/cctv/cameras")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["total"] == 16
    cameras = data["cameras"]
    assert len(cameras) == 16
    assert cameras[0]["camera_id"] == "CAM-001"
    assert "rtmp" in cameras[0]
    assert "location" in cameras[0]


def test_suspect_intake_storage_and_biometrics(client):
    """Test suspect registration with storage and automatic multi-modal feature extraction."""
    payload = {
        "name": "Suresh alias 'Phantom'",
        "alias": "Phantom",
        "fir_no": "FIR-2026-AP-0988",
        "police_station": "PS-KAKINADA-PORT",
        "acts_sec": "BNS Section 303(2), Section 111",
        "known_height_cm": 178.0,
        "torso_leg_ratio": 0.86,
        "stride_length_cm": 68.0,
        "posture_lean_angle": 3.8,
        "clothing_upper_color": "#121826",
        "clothing_lower_color": "#1f2937",
        "carried_objects": ["backpack"],
        "brief_facts": "Suspect in port corridor theft incident."
    }

    res = client.post("/api/suspect/register", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "suspect_id" in data
    assert data["fir_no"] == "FIR-2026-AP-0988"
    assert data["biometrics_extracted"]["known_height_cm"] == 178.0
    assert "backpack" in data["biometrics_extracted"]["carried_objects"]

    # Verify persistent storage in repository
    repo = Repository()
    record = repo.get_criminal_record_by_id(data["suspect_id"])
    assert record is not None
    assert record.accused_name == "Suresh alias 'Phantom'"
    assert record.known_height_cm == 178.0
    assert "backpack" in record.carried_objects


def test_live_cross_camera_search_face_occluded(client):
    """Test live cross-camera search when suspect face is occluded/masked."""
    # Register target suspect
    intake = {
        "name": "Target Occluded",
        "fir_no": "FIR-2026-TEST-OCCLUDED",
        "police_station": "PS-TEST",
        "known_height_cm": 176.0,
        "clothing_upper_color": "#101520",
        "clothing_lower_color": "#202530",
        "carried_objects": ["backpack"]
    }
    reg_res = client.post("/api/suspect/register", json=intake)
    suspect_id = reg_res.json()["suspect_id"]

    # Search live feeds
    res = client.post("/api/suspect/search_live", json={"suspect_id": suspect_id, "min_confidence": 0.40})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["matches_count"] >= 1
    best_candidate = data["candidates"][0]
    assert "alert_id" in best_candidate
    assert "scores" in best_candidate
    assert "biometric_comparison" in best_candidate


def test_officer_confirmation_and_bsa_certificate(client):
    """Test Human-in-the-Loop (HITL) officer confirmation gate and Section 63 BSA certificate."""
    payload = {
        "alert_id": "ALT-2026-TEST-9988",
        "decision": "CONFIRMED_MATCH",
        "officer_name": "Inspector K. Surya",
        "officer_badge": "AP-EG-7744",
        "notes": "Facial symmetry, height 176cm, and black dual-strap backpack verified."
    }

    res = client.post("/api/alerts/officer_confirm", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["decision"] == "CONFIRMED_DISPATCH"
    assert "certificate_digest" in data
    assert len(data["certificate_digest"]) > 10


def test_super_resolution_enhancement_endpoint(client):
    """Test automatic crop super-resolution enhancement."""
    # Create small test crop
    img = np.full((64, 32, 3), 120, dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', img)
    import base64
    b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

    res = client.post("/api/enhancement/enhance_crop", json={"crop_base64": b64, "scale": 2.0})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "enhanced_image_base64" in data
    assert "sha256_hash" in data


def test_dashboard_cameras_and_mode_endpoints(client):
    """Verify official dashboard /cameras, /get_mode, and /set_mode endpoints."""
    # Test /cameras endpoint
    res = client.get("/cameras")
    assert res.status_code == 200
    cams = res.json()
    assert len(cams) == 16
    assert cams[0]["id"] == "CAM-001"
    assert cams[0]["status"] == "online"
    assert "people_count" in cams[0]
    assert "forecast" in cams[0]
    assert cams[0]["forecast"]["status"] == "active"

    # Test /get_mode and /set_mode
    mode_res = client.get("/get_mode")
    assert mode_res.status_code == 200
    assert "counting_mode" in mode_res.json()

    set_res = client.post("/set_mode", json={"counting_mode": "viewing"})
    assert set_res.status_code == 200
    assert set_res.json()["counting_mode"] == "viewing"

    # Reset back to counting
    client.post("/set_mode", json={"counting_mode": "counting"})


def test_dashboard_notifications_and_csv_export(client):
    """Verify /api/notifications and /forecast/history.csv downloads."""
    # Test notifications
    res = client.get("/api/notifications")
    assert res.status_code == 200
    notifs = res.json()
    assert isinstance(notifs, list)
    assert len(notifs) >= 1
    assert "message" in notifs[0]
    assert "severity" in notifs[0]

    # Test CSV export
    csv_res = client.get("/forecast/history.csv")
    assert csv_res.status_code == 200
    assert "text/csv" in csv_res.headers.get("content-type", "")
    content = csv_res.text
    assert "timestamp,camera_id,camera_name" in content
    assert "CAM-001" in content

