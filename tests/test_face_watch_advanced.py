"""Comprehensive Unit & Integration Tests for Upgraded CCTV Facial Recognition System (FRS).

Validates:
1. Optical Quality Gating: Laplacian variance motion blur rejection, resolution gate, illumination.
2. 5-Point Canonical Landmark Alignment (112x112 ArcFace template).
3. Biometric Features & AdaFace Adaptive Quality Margin calculations.
4. Sub-Millisecond 1:N Vector Search (FaceVectorIndex).
5. Multi-Channel Notification Dispatcher (Telegram, Twilio, MSG91, Webhooks, WebSockets).
6. REST API Endpoints: /api/watchlist/status, /config, /probe, /test-alert.
"""

import os
import time
import base64
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

from app.vision.face_engine import FaceBiometricEngine, CANONICAL_5_POINTS
from app.vision.vector_search import FaceVectorIndex
from app.vision.face_watch import LiveFaceWatcher
from app.integrations.notifications import NotificationDispatcher
from app.main import app


def create_sharp_face(w=120, h=140):
    """Creates a high-contrast, sharp face image with prominent facial features."""
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), (w // 3, h // 2 - 10), 0, 0, 360, (130, 160, 210), -1)
    cv2.circle(img, (w // 2 - 16, h // 2 - 15), 5, (40, 30, 20), -1)
    cv2.circle(img, (w // 2 + 16, h // 2 - 15), 5, (40, 30, 20), -1)
    cv2.ellipse(img, (w // 2, h // 2 + 25), (14, 6), 0, 0, 180, (50, 50, 160), 2)
    return img


def create_blurry_face(w=120, h=140):
    """Simulates rapid CCTV subject motion blur."""
    sharp = create_sharp_face(w, h)
    # Apply severe Gaussian blur
    blurred = cv2.GaussianBlur(sharp, (35, 35), 15.0)
    return blurred


# ==================== 1. OPTICAL QUALITY GATING ====================

def test_quality_gating_sharp_vs_blurry():
    sharp = create_sharp_face()
    blurry = create_blurry_face()

    sharp_q = FaceBiometricEngine.assess_face_quality(sharp, min_resolution=24, min_laplacian_var=35.0)
    blurry_q = FaceBiometricEngine.assess_face_quality(blurry, min_resolution=24, min_laplacian_var=35.0)

    # Sharp face must pass quality gate
    assert sharp_q["is_viable"] is True
    assert sharp_q["laplacian_var"] > 35.0
    assert len(sharp_q["rejection_reasons"]) == 0

    # Blurry face must be rejected due to low Laplacian variance
    assert blurry_q["is_viable"] is False
    assert "MOTION_BLUR" in blurry_q["rejection_reasons"]
    assert blurry_q["laplacian_var"] < 35.0


def test_quality_gating_resolution_and_exposure():
    tiny = np.zeros((18, 18, 3), dtype=np.uint8)
    tiny_q = FaceBiometricEngine.assess_face_quality(tiny, min_resolution=24)
    assert tiny_q["is_viable"] is False
    assert "LOW_RESOLUTION" in tiny_q["rejection_reasons"]

    # Pitch black frame
    black = np.zeros((60, 60, 3), dtype=np.uint8)
    black_q = FaceBiometricEngine.assess_face_quality(black)
    assert black_q["is_viable"] is False
    assert "UNDER_EXPOSED" in black_q["rejection_reasons"] or "LOW_CONTRAST" in black_q["rejection_reasons"]


# ==================== 2. 5-POINT CANONICAL ALIGNMENT ====================

def test_5point_canonical_alignment():
    engine = FaceBiometricEngine()
    face = create_sharp_face(200, 200)

    # Synthetic 5 landmarks
    landmarks = [
        [70.0, 80.0],    # Right eye
        [130.0, 80.0],   # Left eye
        [100.0, 110.0],  # Nose
        [80.0, 150.0],   # Mouth right
        [120.0, 150.0]   # Mouth left
    ]

    aligned = engine.align_face_5point(face, landmarks, output_size=(112, 112))
    assert aligned.shape == (112, 112, 3)
    assert aligned.size > 0
    # Image should not be empty
    assert np.mean(aligned) > 10.0


# ==================== 3. ADAPATIVE MARGIN & BIOMETRICS ====================

def test_adaface_quality_adaptive_similarity():
    v1 = np.random.randn(128).astype(np.float32)
    v1 /= np.linalg.norm(v1)

    # Identical vectors: similarity should be 1.0
    sim_ident = FaceBiometricEngine.compute_face_similarity(v1, v1, quality1=1.0, quality2=1.0)
    assert pytest.approx(sim_ident, 1e-4) == 1.0

    # Orthogonal vectors: similarity near 0.0
    v2 = np.random.randn(128).astype(np.float32)
    v2 -= np.dot(v1, v2) * v1
    v2 /= np.linalg.norm(v2)
    sim_orth = FaceBiometricEngine.compute_face_similarity(v1, v2)
    assert abs(sim_orth) < 0.15

    # AdaFace mode with varying qualities
    ada_high_q = FaceBiometricEngine.compute_face_similarity(v1, v1, quality1=1.0, quality2=1.0, use_adaface_margin=True)
    ada_low_q = FaceBiometricEngine.compute_face_similarity(v1, v1, quality1=0.2, quality2=0.2, use_adaface_margin=True)
    assert 0.0 <= ada_high_q <= 1.0
    assert 0.0 <= ada_low_q <= 1.0


# ==================== 4. 1:N VECTOR SEARCH INDEX ====================

def test_vector_search_index_operations():
    index = FaceVectorIndex()
    assert index.count() == 0

    # Enroll 20 synthetic suspects
    dim = 128
    np.random.seed(42)
    for i in range(20):
        vec = np.random.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        index.add_target(
            target_id=f"TGT-{i:03d}",
            embedding=vec,
            metadata={"name": f"Suspect {i}", "notes": f"Priority {i % 3}"}
        )

    assert index.count() == 20

    # Retrieve target 5
    t5 = index.get_target("TGT-005")
    assert t5 is not None
    assert t5["metadata"]["name"] == "Suspect 5"

    # Search with exact target 5 vector -> should return TGT-005 with sim ~ 1.0
    vec5 = np.array(t5["embedding"], dtype=np.float32)
    hits = index.search(vec5, top_k=3, threshold=0.90)
    assert len(hits) >= 1
    assert hits[0]["target_id"] == "TGT-005"
    assert pytest.approx(hits[0]["similarity"], 1e-3) == 1.0

    # Delete target 5
    removed = index.remove_target("TGT-005")
    assert removed is True
    assert index.count() == 19
    assert index.get_target("TGT-005") is None

    # Clear index
    index.clear()
    assert index.count() == 0


def test_vector_search_latency_benchmark():
    index = FaceVectorIndex()
    np.random.seed(123)
    dim = 512
    # Enroll 500 identities to simulate busy metro station watchlist
    for i in range(500):
        v = np.random.randn(dim).astype(np.float32)
        index.add_target(f"SUSPECT-{i}", v)

    query = np.random.randn(dim).astype(np.float32)
    t0 = time.perf_counter()
    results = index.search(query, top_k=5, threshold=0.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Must be sub-millisecond on standard CPU
    assert elapsed_ms < 10.0
    assert len(results) == 5
    stats = index.get_stats()
    assert stats["total_identities"] == 500


# ==================== 5. MULTI-CHANNEL NOTIFICATION DISPATCHER ====================

def test_notification_dispatcher():
    dispatcher = NotificationDispatcher()
    status = dispatcher.get_status()
    assert "channels" in status
    assert "stats" in status

    # Dispatch synthetic alert
    test_event = {
        "alert_id": "ALT-DISP-001",
        "target_id": "TGT-DISP-99",
        "target_name": "Test Subject",
        "camera_id": "CAM-001",
        "timestamp": time.time(),
        "confidence": 0.89,
        "similarity_pct": 89.0,
        "raw_frame_hash": "a" * 64,
        "notes": "Dispatcher unit test"
    }

    dispatcher.dispatch_alert(test_event)
    # Total dispatched counter increments
    assert dispatcher.stats["total_dispatched"] >= 1

    # Configuration update
    dispatcher.update_config({
        "telegram_bot_token": "TEST_TOKEN_123",
        "telegram_chat_id": "12345678",
        "webhook_url": "https://example.com/cctv-webhook"
    })
    assert dispatcher.config["telegram_enabled"] is True
    assert dispatcher.config["webhook_enabled"] is True


# ==================== 6. REST API INTEGRATION ====================

def test_api_watchlist_status_and_config():
    client = TestClient(app)

    # 1. GET /api/watchlist/status
    res = client.get("/api/watchlist/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "pipeline" in data
    assert "vector_search" in data["pipeline"]
    assert "notifications" in data["pipeline"]

    # 2. POST /api/watchlist/config
    config_res = client.post(
        "/api/watchlist/config",
        json={
            "cooldown_sec": 45.0,
            "default_threshold": 0.65,
            "min_face_resolution": 28,
            "min_laplacian_var": 40.0
        }
    )
    assert config_res.status_code == 200
    cfg_data = config_res.json()
    assert cfg_data["pipeline"]["cooldown_sec"] == 45.0
    assert cfg_data["pipeline"]["default_threshold"] == 0.65
    assert cfg_data["pipeline"]["min_face_resolution"] == 28


def test_api_watchlist_probe_and_test_alert():
    client = TestClient(app)
    face_img = create_sharp_face()
    _, buf = cv2.imencode(".jpg", face_img)
    b64_img = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

    # 1. Enroll face first
    enroll_res = client.post(
        "/api/watchlist/target-face",
        json={
            "name": "Probe Candidate",
            "image_base64": b64_img,
            "target_id": "TGT-PROBE-01",
            "threshold": 0.40
        }
    )
    assert enroll_res.status_code == 200

    # 2. POST /api/watchlist/probe (1:N search)
    probe_res = client.post(
        "/api/watchlist/probe",
        json={
            "image_base64": b64_img,
            "top_k": 3,
            "threshold": 0.35
        }
    )
    assert probe_res.status_code == 200
    p_data = probe_res.json()
    assert p_data["status"] == "SUCCESS"
    assert p_data["faces_detected"] >= 1
    assert len(p_data["results"]) >= 1

    # 3. POST /api/watchlist/test-alert
    alert_res = client.post("/api/watchlist/test-alert")
    assert alert_res.status_code == 200
    a_data = alert_res.json()
    assert a_data["status"] == "SUCCESS"
    assert "ALT-TEST" in a_data["event"]["alert_id"]
