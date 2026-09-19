"""Unit tests for Model Downloader and Person Evidence Packet."""

import pytest
import os
from fastapi.testclient import TestClient

from app.main import app
from app.adapters.downloader import model_downloader, GITHUB_MODEL_CATALOG
from app.evidence.packet import PersonEvidencePacket


@pytest.fixture
def client():
    return TestClient(app)


def test_github_model_catalog_integrity():
    """Verify that all catalog entries have valid URLs, categories, and filenames."""
    assert len(GITHUB_MODEL_CATALOG) >= 5
    for key, info in GITHUB_MODEL_CATALOG.items():
        assert info["model_id"] == key
        assert info["url"].startswith("http")
        assert info["filename"]
        assert info["size_bytes"] > 1000
        assert info["category"] in ["face", "detection", "pose", "reid", "gait", "tracking"]


def test_model_downloader_status():
    """Verify get_status returns accurate dictionary."""
    status = model_downloader.get_status()
    assert isinstance(status, dict)
    assert "face_yunet" in status
    assert "yolov8n" in status
    assert "osnet" in status

    # Essential models that we downloaded should be marked downloaded
    assert status["face_yunet"]["is_downloaded"] is True
    assert status["yolov8n"]["is_downloaded"] is True
    assert status["osnet"]["is_downloaded"] is True


def test_person_evidence_packet_face_not_gatekeeper():
    """Verify that when face is unavailable or masked, other evidence remains fully accessible."""
    packet = PersonEvidencePacket(
        track_id="CAM017-T481",
        camera_id="CAM-017",
        body_embedding=[0.1] * 512,
        gait_embedding=[0.2] * 32,
        clothing={"clothing_upper": "#223344", "clothing_lower": "#000000"},
        height_estimate=178.5,
        object_attributes=["backpack", "black_cap"],
        trajectory=[(100.0, 200.0, 1.0), (120.0, 210.0, 2.0)],
        quality={
            "face": 0.0,  # FACE 0% VISIBLE (Rear view / Helmet)
            "body": 0.90,
            "gait": 0.85,
            "pose": 0.80,
            "height": 0.92,
            "clothing": 0.88,
            "trajectory": 0.95
        }
    )

    # Crucial architectural requirement: Face is NOT available, but packet is valid!
    assert packet.is_face_available() is False
    avail_mods = packet.get_available_modalities()
    assert "face" not in avail_mods
    assert "body" in avail_mods
    assert "gait" in avail_mods
    assert "clothing" in avail_mods
    assert "height" in avail_mods
    assert "carried_objects" in avail_mods
    assert "trajectory" in avail_mods

    p_dict = packet.to_dict()
    assert p_dict["face_available"] is False
    assert p_dict["body_embedding_len"] == 512
    assert p_dict["height_estimate"] == 178.5


def test_api_download_status_endpoint(client):
    """Test the /api/models/download-status endpoint."""
    resp = client.get("/api/models/download-status")
    assert resp.status_code == 200
    data = resp.json()
    assert "yolov8n" in data
    assert data["yolov8n"]["is_downloaded"] is True
