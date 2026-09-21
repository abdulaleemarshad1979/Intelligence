"""Unit & Integration tests for Ultra-Low-Latency CCTV Stream Optimization and Matrix Camera Integration."""

import time
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from app.main import app, stream_mgr
from app.ingestion.stream_manager import (
    build_matrix_rtsp_url,
    CameraStreamWorker,
    CameraStreamManager
)
from app.features.enhancement import stream_enhancer


@pytest.fixture
def client():
    return TestClient(app)


def test_build_matrix_rtsp_url_variants():
    """Verify Matrix Comsec RTSP stream URL synthesis for all profiles."""
    # 1. Main stream high resolution (1080p / 4MP)
    url_main = build_matrix_rtsp_url(
        ip="192.168.1.130",
        port=554,
        username="admin",
        password="password123",
        stream_type="media/video1"
    )
    assert url_main == "rtsp://admin:password123@192.168.1.130:554/media/video1"

    # 2. Sub stream low latency
    url_sub = build_matrix_rtsp_url(
        ip="192.168.1.130",
        port=554,
        username="admin",
        password="password123",
        stream_type="media/video2"
    )
    assert url_sub == "rtsp://admin:password123@192.168.1.130:554/media/video2"

    # 3. Live channel 1
    url_live = build_matrix_rtsp_url(
        ip="10.0.0.50",
        port=554,
        stream_type="live1"
    )
    assert url_live == "rtsp://10.0.0.50:554/live1"


def test_fast_enhancer_speed_and_quality():
    """Ensure FastEnhancer executes with high speed on full 1024x576 CCTV frames."""
    frame = np.random.randint(0, 255, (576, 1024, 3), dtype=np.uint8)
    
    # Warmup
    _ = stream_enhancer.enhance_stream_frame(frame)
    
    t0 = time.perf_counter()
    enhanced = stream_enhancer.enhance_stream_frame(frame)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert enhanced.shape == frame.shape
    assert elapsed_ms < 35.0, f"Enhancer too slow: {elapsed_ms:.2f}ms (must be < 35ms)"


def test_camera_stream_worker_lifecycle():
    """Verify CameraStreamWorker captures frames and produces rendered frames."""
    worker = CameraStreamWorker(
        camera_id="CAM-TEST-001",
        source="simulated://test",
        name="Test Camera",
        target_fps=25,
        enable_ai=False
    )
    worker.start()
    assert worker.is_running is True

    # Wait briefly for first frame
    time.sleep(0.1)
    frame = worker.get_latest_rendered_frame(overlay_mode="clean")
    assert frame is not None
    assert frame.shape[0] > 0
    assert frame.shape[1] > 0

    stats = worker.get_stats()
    assert stats["camera_id"] == "CAM-TEST-001"
    assert stats["is_running"] is True
    assert stats["fps"] > 0

    worker.stop()
    assert worker.is_running is False


def test_api_connect_matrix_camera(client):
    """Test connecting a Matrix CCTV camera via POST /api/cameras/connect_matrix."""
    payload = {
        "camera_id": "CAM-002",
        "ip": "192.168.1.135",
        "port": 554,
        "username": "admin",
        "password": "MatrixAdminPassword",
        "stream_type": "media/video1",
        "name": "Matrix Hospital Ambulance Bay"
    }

    res = client.post("/api/cameras/connect_matrix", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["camera_id"] == "CAM-002"
    assert "media/video1" in data["rtsp_url"]
    assert "admin:MatrixAdminPassword" in data["rtsp_url"]
    assert "stream_info" in data


def test_api_camera_stream_info(client):
    """Test retrieving live telemetry from GET /api/cameras/{camera_id}/stream_info."""
    res = client.get("/api/cameras/CAM-001/stream_info")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    info = data["stream_info"]
    assert info["camera_id"] == "CAM-001"
    assert "fps" in info
    assert "latency_ms" in info


def test_mjpeg_stream_generator():
    """Verify stream manager produces valid multipart MJPEG frame chunks."""
    gen = stream_mgr.generate_mjpeg_stream(camera_id="CAM-001", quality=85)
    try:
        chunk = next(gen)
        assert len(chunk) > 0
        assert b"--frame" in chunk
        assert b"Content-Type: image/jpeg" in chunk
    finally:
        gen.close()
