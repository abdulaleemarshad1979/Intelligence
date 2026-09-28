"""Unit and integration tests for Multi-Camera Target Person Search.

Covers:
- Reference image validation (empty, low-res, valid)
- Session creation & automatic mode resolution (face vs appearance Re-ID fallback)
- Single-frame rejection (single detection cannot confirm)
- Multi-frame temporal confirmation across rolling window
- Rolling-window expiration of stale observations
- Duplicate event suppression during cooldown
- Multi-camera search coordination
- Camera failure isolation (broken camera does not terminate session)
- Best evidence frame selection & SHA-256 cryptographic verification
- No artificial confidence mapping (exposes raw similarity, quality, confirmation score)
- REST API workflows: start, status, events, cameras, evidence, stop
"""

import os
import io
import time
import json
import hashlib
import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.target_search.models import (
    Observation, CandidateEvent, CandidateStatus, SessionStatus,
    TargetSearchConfig, TargetSearchSession
)
from app.target_search.confirmation import (
    CandidateTrackState, TemporalConfirmationEngine, compute_sha256
)
from app.target_search.coordinator import TargetSearchCoordinator
from app.main import app


def create_synthetic_face_portrait(w=120, h=140, skin_color=(130, 160, 210)) -> np.ndarray:
    """Create a synthetic face image with recognizable facial geometry."""
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), (w // 3, h // 2 - 10), 0, 0, 360, skin_color, -1)
    cv2.circle(img, (w // 2 - 16, h // 2 - 15), 5, (40, 30, 20), -1)
    cv2.circle(img, (w // 2 + 16, h // 2 - 15), 5, (40, 30, 20), -1)
    cv2.ellipse(img, (w // 2, h // 2 + 25), (14, 6), 0, 0, 180, (50, 50, 160), 2)
    return img


def create_synthetic_person_body(w=80, h=180) -> np.ndarray:
    """Create a synthetic full body silhouette without a clear facial structure."""
    img = np.full((h, w, 3), 50, dtype=np.uint8)
    # Head / hair (dark, rear angle)
    cv2.circle(img, (w // 2, 25), 18, (20, 20, 20), -1)
    # Torso (blue jacket)
    cv2.rectangle(img, (15, 45), (w - 15, 110), (140, 70, 30), -1)
    # Legs (dark pants)
    cv2.rectangle(img, (20, 110), (w // 2 - 4, h - 10), (40, 40, 40), -1)
    cv2.rectangle(img, (w // 2 + 4, 110), (w - 20, h - 10), (40, 40, 40), -1)
    return img


# ==================== 1. REFERENCE IMAGE VALIDATION ====================

def test_reference_validation_empty_or_corrupt(tmp_path):
    coord = TargetSearchCoordinator(base_dir=str(tmp_path))

    with pytest.raises(ValueError, match="empty or unreadable"):
        coord.decode_and_validate_reference(b"")

    with pytest.raises(ValueError, match="empty or unreadable"):
        coord.decode_and_validate_reference(b"not an image file content")


def test_reference_validation_low_resolution(tmp_path):
    coord = TargetSearchCoordinator(base_dir=str(tmp_path))
    tiny_img = np.zeros((16, 16, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", tiny_img)

    with pytest.raises(ValueError, match="too low"):
        coord.decode_and_validate_reference(buf.tobytes())


def test_reference_validation_valid(tmp_path):
    coord = TargetSearchCoordinator(base_dir=str(tmp_path))
    valid_img = create_synthetic_face_portrait(120, 140)
    decoded = coord.decode_and_validate_reference(valid_img)
    assert decoded is not None
    assert decoded.shape == (140, 120, 3)


# ==================== 2. SESSION CREATION & MODE RESOLUTION ====================

def test_session_creation_auto_mode_with_face(tmp_path):
    coord = TargetSearchCoordinator(base_dir=str(tmp_path))
    face_img = create_synthetic_face_portrait()

    res = coord.start_search(
        image_input=face_img,
        name="Subject Alpha",
        mode="auto",
        cameras="CAM-001,CAM-002"
    )

    assert res["status"] == "SUCCESS"
    assert res["session_id"].startswith("TS-")
    assert res["selected_cameras"] == ["CAM-001", "CAM-002"]

    session = coord.get_session(res["session_id"])
    assert session is not None
    assert session.active_mode == "face"
    assert session.reference_face_embedding is not None

    coord.stop_search(res["session_id"])


def test_session_creation_auto_mode_fallback_to_reid(tmp_path):
    coord = TargetSearchCoordinator(base_dir=str(tmp_path))
    # Body without clear face
    body_img = create_synthetic_person_body()

    # Mock face detector to simulate no detectable face
    with patch.object(coord.face_engine, "detect_faces", return_value=[]):
        res = coord.start_search(
            image_input=body_img,
            name="Subject Rear View",
            mode="auto",
            cameras="CAM-003"
        )

        assert res["status"] == "SUCCESS"
        session = coord.get_session(res["session_id"])
        assert session is not None
        assert session.active_mode == "reid"
        assert session.reference_reid_embedding is not None

        coord.stop_search(res["session_id"])


# ==================== 3. SINGLE FRAME REJECTION ====================

def test_single_frame_rejection(tmp_path):
    cfg = TargetSearchConfig(min_confirmations=4, confirmation_window_sec=8.0)
    engine = TemporalConfirmationEngine(
        session_id="TS-TEST-001",
        storage_base_dir=str(tmp_path),
        config=cfg
    )

    dummy_frame = np.full((360, 640, 3), 50, dtype=np.uint8)
    dummy_crop = np.full((120, 60, 3), 80, dtype=np.uint8)

    obs = Observation(
        timestamp=time.time(),
        camera_id="CAM-001",
        track_id="17",
        bbox=[100, 100, 60, 120],
        raw_similarity=0.88,
        quality_score=0.85,
        detection_score=0.92,
        frame_id=1
    )

    track_state, event = engine.process_observation(obs, dummy_frame, dummy_crop, None)

    # 1 single observation MUST NOT confirm candidate
    assert event is None
    assert track_state.status == CandidateStatus.OBSERVED
    assert track_state.confirmation_count == 1
    assert len(engine.get_events()) == 0


# ==================== 4. MULTI-FRAME CONFIRMATION & SCORING ====================

def test_multi_frame_confirmation_and_scoring(tmp_path):
    cfg = TargetSearchConfig(
        min_confirmations=4,
        confirmation_window_sec=8.0,
        minimum_raw_similarity=0.55,
        minimum_quality_score=0.35,
        duplicate_event_cooldown_sec=15.0
    )
    engine = TemporalConfirmationEngine(
        session_id="TS-CONFIRM-001",
        storage_base_dir=str(tmp_path),
        config=cfg
    )

    dummy_frame = np.full((360, 640, 3), 40, dtype=np.uint8)
    dummy_crop = np.full((140, 70, 3), 90, dtype=np.uint8)
    dummy_face = np.full((50, 50, 3), 120, dtype=np.uint8)

    base_time = time.time()
    sims = [0.65, 0.72, 0.78, 0.81]
    quals = [0.80, 0.82, 0.85, 0.84]

    event = None
    track_state = None

    for idx, (s, q) in enumerate(zip(sims, quals)):
        obs = Observation(
            timestamp=base_time + idx * 0.5,
            camera_id="CAM-002",
            track_id="27",
            bbox=[150, 80, 70, 140],
            raw_similarity=s,
            quality_score=q,
            detection_score=0.90,
            frame_id=100 + idx
        )
        track_state, event = engine.process_observation(obs, dummy_frame, dummy_crop, dummy_face)

    # After 4 valid observations, candidate must be confirmed
    assert track_state.status == CandidateStatus.CONFIRMED_CANDIDATE
    assert track_state.confirmation_count == 4
    assert event is not None
    assert event.event_id.startswith("EVT-TS-")
    assert event.camera_id == "CAM-002"
    assert event.track_id == "27"

    # Verify score is clamped between 0 and 1
    assert 0.0 <= event.confirmation_score <= 1.0

    # Verify raw metrics exposure without fake conversion
    assert event.raw_similarity_max == round(max(sims), 4)
    assert event.raw_similarity_mean == round(float(np.mean(sims)), 4)
    assert event.quality_mean == round(float(np.mean(quals)), 4)
    assert event.review_required is True


# ==================== 5. ROLLING WINDOW EXPIRATION ====================

def test_rolling_window_expiration(tmp_path):
    cfg = TargetSearchConfig(
        min_confirmations=3,
        confirmation_window_sec=5.0,
        minimum_raw_similarity=0.50,
        minimum_quality_score=0.30
    )
    engine = TemporalConfirmationEngine(
        session_id="TS-EXPIRE-001",
        storage_base_dir=str(tmp_path),
        config=cfg
    )

    t0 = 1000.0
    dummy_frame = np.full((100, 100, 3), 30, dtype=np.uint8)

    # First observation at t0
    obs1 = Observation(t0, "CAM-001", "5", [10, 10, 30, 60], 0.70, 0.80, 0.90, 1)
    engine.process_observation(obs1, dummy_frame)

    # Second observation 2s later
    obs2 = Observation(t0 + 2.0, "CAM-001", "5", [10, 10, 30, 60], 0.72, 0.80, 0.90, 2)
    engine.process_observation(obs2, dummy_frame)

    # Third observation 10s later (obs1 and obs2 are now older than 5s window!)
    obs3 = Observation(t0 + 10.0, "CAM-001", "5", [10, 10, 30, 60], 0.75, 0.80, 0.90, 3)
    track_state, event = engine.process_observation(obs3, dummy_frame)

    # Must NOT confirm because obs1 and obs2 expired from rolling window
    assert event is None
    assert track_state.confirmation_count == 1  # Only obs3 is in window
    assert track_state.status != CandidateStatus.CONFIRMED_CANDIDATE


# ==================== 6. DUPLICATE EVENT SUPPRESSION ====================

def test_duplicate_event_cooldown_suppression(tmp_path):
    cfg = TargetSearchConfig(
        min_confirmations=2,
        confirmation_window_sec=10.0,
        duplicate_event_cooldown_sec=15.0
    )
    engine = TemporalConfirmationEngine(
        session_id="TS-DUP-001",
        storage_base_dir=str(tmp_path),
        config=cfg
    )

    dummy_frame = np.full((100, 100, 3), 30, dtype=np.uint8)
    t = time.time()

    # Observation 1
    engine.process_observation(Observation(t, "CAM-001", "9", [0, 0, 40, 80], 0.75, 0.8, 0.9, 1), dummy_frame)
    # Observation 2 -> Triggers Event 1
    _, e1 = engine.process_observation(Observation(t + 1, "CAM-001", "9", [0, 0, 40, 80], 0.78, 0.8, 0.9, 2), dummy_frame)
    assert e1 is not None

    # Observation 3 at t + 3s (within 15s cooldown) -> Must NOT emit duplicate event
    _, e2 = engine.process_observation(Observation(t + 3, "CAM-001", "9", [0, 0, 40, 80], 0.80, 0.8, 0.9, 3), dummy_frame)
    assert e2 is None

    # Total events stored remains 1
    assert len(engine.get_events()) == 1


# ==================== 7. MULTI-CAMERA & CAMERA FAILURE ISOLATION ====================

def test_multi_camera_isolation(tmp_path):
    coord = TargetSearchCoordinator(base_dir=str(tmp_path))
    face_img = create_synthetic_face_portrait()

    res = coord.start_search(
        image_input=face_img,
        name="Target MultiCam",
        mode="auto",
        cameras=["CAM-001", "CAM-002", "CAM-003"]
    )
    session_id = res["session_id"]

    # Verify session has all 3 cameras
    status = coord.get_status(session_id)
    assert status["active_cameras_count"] == 3

    # Check cameras status endpoint
    cams = coord.get_cameras_status(session_id)
    assert len(cams) == 3
    cam_ids = [c["camera_id"] for c in cams]
    assert "CAM-001" in cam_ids
    assert "CAM-002" in cam_ids
    assert "CAM-003" in cam_ids

    # Simulate one broken camera: worker fails to grab frame or throws
    broken_worker = coord.stream_manager.get_or_create_worker("CAM-003")
    with patch.object(broken_worker, "get_latest_frame", side_effect=RuntimeError("RTSP stream disconnected")):
        # Let background loop iterate
        time.sleep(0.3)
        # Search session must still be running
        assert coord.get_session(session_id).status == SessionStatus.RUNNING.value

    coord.stop_search(session_id)


# ==================== 8. BEST EVIDENCE & SHA-256 INTEGRITY ====================

def test_best_evidence_and_sha256(tmp_path):
    cfg = TargetSearchConfig(min_confirmations=2, confirmation_window_sec=8.0)
    engine = TemporalConfirmationEngine(
        session_id="TS-EVID-001",
        storage_base_dir=str(tmp_path),
        config=cfg
    )

    frame1 = np.full((200, 300, 3), 40, dtype=np.uint8)
    crop1 = np.full((80, 40, 3), 60, dtype=np.uint8)
    face1 = np.full((30, 30, 3), 100, dtype=np.uint8)

    t = time.time()
    # Frame 1: Low quality (0.4)
    engine.process_observation(Observation(t, "CAM-001", "44", [10, 10, 40, 80], 0.60, 0.40, 0.70, 1), frame1, crop1, face1)

    # Frame 2: High quality (0.9) -> Triggers confirmation and selects Frame 2 as best frame
    frame2 = np.full((200, 300, 3), 200, dtype=np.uint8)
    crop2 = np.full((80, 40, 3), 220, dtype=np.uint8)
    face2 = np.full((30, 30, 3), 250, dtype=np.uint8)

    _, event = engine.process_observation(
        Observation(t + 0.5, "CAM-001", "44", [10, 10, 40, 80], 0.85, 0.90, 0.95, 2),
        frame2, crop2, face2
    )

    assert event is not None
    event_dir = os.path.join(str(tmp_path), "TS-EVID-001", event.event_id)
    assert os.path.isdir(event_dir)

    full_path = os.path.join(event_dir, "full.jpg")
    person_path = os.path.join(event_dir, "person.jpg")
    face_path = os.path.join(event_dir, "face.jpg")
    meta_path = os.path.join(event_dir, "metadata.json")
    hashes_path = os.path.join(event_dir, "hashes.json")

    assert os.path.isfile(full_path)
    assert os.path.isfile(person_path)
    assert os.path.isfile(face_path)
    assert os.path.isfile(meta_path)
    assert os.path.isfile(hashes_path)

    # Verify SHA-256 hashes
    with open(hashes_path, "r", encoding="utf-8") as f:
        stored_hashes = json.load(f)

    with open(full_path, "rb") as f:
        assert compute_sha256(f.read()) == stored_hashes["full.jpg"]

    with open(person_path, "rb") as f:
        assert compute_sha256(f.read()) == stored_hashes["person.jpg"]

    with open(face_path, "rb") as f:
        assert compute_sha256(f.read()) == stored_hashes["face.jpg"]


# ==================== 9. REST API ENDPOINTS ====================

def test_api_target_search_lifecycle():
    client = TestClient(app)

    face_img = create_synthetic_face_portrait(120, 140)
    _, buf = cv2.imencode(".jpg", face_img)
    img_bytes = io.BytesIO(buf.tobytes())

    # 1. Start search
    response = client.post(
        "/api/target-search/start",
        files={"image": ("test_person.jpg", img_bytes, "image/jpeg")},
        data={"name": "Suspect John Doe", "mode": "auto", "cameras": "CAM-001,CAM-002"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    session_id = data["session_id"]
    assert "selected_cameras" in data
    assert len(data["selected_cameras"]) == 2

    # 2. Get status
    status_resp = client.get(f"/api/target-search/{session_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["session_id"] == session_id
    assert status_data["status"] == "Running"
    assert status_data["active_cameras_count"] == 2

    # 3. Get events
    events_resp = client.get(f"/api/target-search/{session_id}/events")
    assert events_resp.status_code == 200
    assert isinstance(events_resp.json(), list)

    # 4. Get cameras status
    cams_resp = client.get(f"/api/target-search/{session_id}/cameras")
    assert cams_resp.status_code == 200
    cams_data = cams_resp.json()
    assert len(cams_data) == 2

    # 5. Get evidence
    evid_resp = client.get(f"/api/target-search/{session_id}/evidence")
    assert evid_resp.status_code == 200
    assert isinstance(evid_resp.json(), list)

    # 6. Stop search
    stop_resp = client.post(f"/api/target-search/{session_id}/stop")
    assert stop_resp.status_code == 200
    stop_data = stop_resp.json()
    assert stop_data["status"] == "SUCCESS"

    # Verify status changed to Stopped
    status_after = client.get(f"/api/target-search/{session_id}").json()
    assert status_after["status"] == "Stopped"


# ==================== 10. NO ARTIFICIAL CONFIDENCE RANGE ====================

def test_no_artificial_confidence_conversion(tmp_path):
    cfg = TargetSearchConfig(min_confirmations=1, minimum_raw_similarity=0.40, minimum_quality_score=0.20)
    engine = TemporalConfirmationEngine(
        session_id="TS-NO-FAKE-001",
        storage_base_dir=str(tmp_path),
        config=cfg
    )

    dummy_frame = np.full((100, 100, 3), 30, dtype=np.uint8)
    raw_sim = 0.52
    quality = 0.65
    det_score = 0.88

    obs = Observation(
        timestamp=time.time(),
        camera_id="CAM-001",
        track_id="3",
        bbox=[10, 10, 40, 80],
        raw_similarity=raw_sim,
        quality_score=quality,
        detection_score=det_score,
        frame_id=1
    )

    track_state, event = engine.process_observation(obs, dummy_frame)

    assert event is not None
    # Must NOT map to artificial 68%-98% range
    # It must expose genuine raw similarity
    assert event.raw_similarity_max == raw_sim
    assert event.raw_similarity_mean == raw_sim
    assert event.quality_mean == quality
    assert event.status == "CONFIRMED_CANDIDATE"
    # Confirmation score must be computed using formula, not mapped artificially
    assert event.confirmation_score != 0.985
    assert event.confirmation_score != 0.68
