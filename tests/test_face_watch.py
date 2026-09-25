"""Unit & Integration tests for Live Face Watcher & Automated Snapshot Capture."""

import os
import time
import base64
import tempfile
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.vision.face_watch import LiveFaceWatcher
from app.main import app


@pytest.fixture
def temp_watcher(tmp_path):
    storage_dir = str(tmp_path / "captures")
    watcher = LiveFaceWatcher(
        storage_dir=storage_dir,
        cooldown_sec=1.0,
        default_threshold=0.50,
        sync_db=False
    )
    watcher.targets_dir = str(tmp_path / "targets")
    os.makedirs(watcher.targets_dir, exist_ok=True)
    return watcher


def create_synthetic_face_image(w=120, h=140, skin_color=(130, 160, 210)):
    """Generates a synthetic portrait image with facial structure (head, eyes, mouth)."""
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    # Head oval
    cv2.ellipse(img, (w // 2, h // 2), (w // 3, h // 2 - 10), 0, 0, 360, skin_color, -1)
    # Eyes
    cv2.circle(img, (w // 2 - 16, h // 2 - 15), 5, (40, 30, 20), -1)
    cv2.circle(img, (w // 2 + 16, h // 2 - 15), 5, (40, 30, 20), -1)
    # Mouth
    cv2.ellipse(img, (w // 2, h // 2 + 25), (14, 6), 0, 0, 180, (50, 50, 160), 2)
    return img


def test_target_enrollment(temp_watcher):
    face_img = create_synthetic_face_image()
    target = temp_watcher.enroll_target_face(
        image_input=face_img,
        name="Target Alpha",
        target_id="TGT-TEST-001",
        threshold=0.55,
        notes="High priority person of interest"
    )

    assert target["target_id"] == "TGT-TEST-001"
    assert target["name"] == "Target Alpha"
    assert target["threshold"] == 0.55
    assert len(target["embedding"]) in (128, 512)
    assert os.path.isfile(target["reference_path"])
    assert "TGT-TEST-001" in temp_watcher.targets


def test_live_frame_match_and_auto_capture(temp_watcher):
    face_img = create_synthetic_face_image()
    temp_watcher.enroll_target_face(
        image_input=face_img,
        name="Target Beta",
        target_id="TGT-BETA",
        threshold=0.45
    )

    # Create synthetic surveillance scene containing the target
    scene = np.full((360, 640, 3), 30, dtype=np.uint8)
    fh, fw = face_img.shape[:2]
    scene[100:100 + fh, 200:200 + fw] = face_img

    # Process frame through live face watcher
    matches = temp_watcher.process_frame(scene, camera_id="CAM-002", frame_id=42)

    assert len(matches) == 1
    match = matches[0]
    assert match["target_id"] == "TGT-BETA"
    assert match["target_name"] == "Target Beta"
    assert match["camera_id"] == "CAM-002"
    assert match["confidence"] >= 0.45
    assert match["similarity_pct"] >= 45.0

    # Verify captured snapshots exist on disk
    assert os.path.isfile(match["full_frame_path"])
    assert os.path.isfile(match["face_crop_path"])
    assert os.path.getsize(match["full_frame_path"]) > 0
    assert os.path.getsize(match["face_crop_path"]) > 0

    # Verify cryptographic SHA-256 hash
    assert len(match["raw_frame_hash"]) == 64
    assert len(match["face_crop_hash"]) == 64


def test_cooldown_debounce(temp_watcher):
    face_img = create_synthetic_face_image()
    temp_watcher.enroll_target_face(
        image_input=face_img,
        name="Target Gamma",
        target_id="TGT-GAMMA",
        threshold=0.40
    )

    scene = np.full((360, 640, 3), 30, dtype=np.uint8)
    fh, fw = face_img.shape[:2]
    scene[50:50 + fh, 50:50 + fw] = face_img

    # First sighting -> triggers capture
    m1 = temp_watcher.process_frame(scene, camera_id="CAM-001", frame_id=1)
    assert len(m1) == 1

    # Immediate second sighting -> suppressed by cooldown
    m2 = temp_watcher.process_frame(scene, camera_id="CAM-001", frame_id=2)
    assert len(m2) == 0

    # Wait for cooldown to expire
    time.sleep(1.1)
    # Third sighting after cooldown -> triggers new capture
    m3 = temp_watcher.process_frame(scene, camera_id="CAM-001", frame_id=3)
    assert len(m3) == 1


def test_non_matching_face_rejected(temp_watcher):
    face_img = create_synthetic_face_image()
    temp_watcher.enroll_target_face(
        image_input=face_img,
        name="Target Delta",
        target_id="TGT-DELTA",
        threshold=0.98  # Very high threshold impossible for noise
    )

    # Blank/noise scene
    empty_scene = np.full((360, 640, 3), 100, dtype=np.uint8)
    matches = temp_watcher.process_frame(empty_scene, camera_id="CAM-001", frame_id=99)
    assert len(matches) == 0


def test_api_watchlist_endpoints():
    client = TestClient(app)
    face_img = create_synthetic_face_image()
    _, buf = cv2.imencode(".jpg", face_img)
    b64_img = base64.b64encode(buf).decode("utf-8")

    # 1. Enroll target face
    enroll_resp = client.post(
        "/api/watchlist/target-face",
        json={
            "name": "API Suspect",
            "image_base64": f"data:image/jpeg;base64,{b64_img}",
            "target_id": "TGT-API-01",
            "threshold": 0.55,
            "notes": "Enrolled via test client"
        }
    )
    assert enroll_resp.status_code == 200
    data = enroll_resp.json()
    assert data["status"] == "SUCCESS"
    assert data["target"]["name"] == "API Suspect"

    # 2. List target faces
    list_resp = client.get("/api/watchlist/target-faces")
    assert list_resp.status_code == 200
    targets = list_resp.json()["targets"]
    assert any(t["target_id"] == "TGT-API-01" for t in targets)

    # 3. List captures
    captures_resp = client.get("/api/watchlist/captures")
    assert captures_resp.status_code == 200
    assert "captures" in captures_resp.json()

    # 4. Delete target face
    del_resp = client.delete("/api/watchlist/target-faces/TGT-API-01")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "SUCCESS"


def test_api_enroll_video_endpoint(tmp_path):
    client = TestClient(app)
    video_path = str(tmp_path / "test_walk.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 15.0, (320, 240))
    for i in range(12):
        frame = np.full((240, 320, 3), 50, dtype=np.uint8)
        # Draw a synthetic moving human
        cx = 100 + i * 5
        cv2.circle(frame, (cx, 60), 15, (200, 200, 200), -1)  # head
        cv2.line(frame, (cx, 75), (cx, 150), (200, 200, 200), 4)  # torso
        cv2.line(frame, (cx, 150), (cx - 15, 210), (200, 200, 200), 4)  # leg 1
        cv2.line(frame, (cx, 150), (cx + 15, 210), (200, 200, 200), 4)  # leg 2
        out.write(frame)
    out.release()

    with open(video_path, "rb") as vf:
        resp = client.post(
            "/api/watchlist/enroll-video",
            files={"video_file": ("test_walk.mp4", vf, "video/mp4")},
            data={"name": "Video Target Test", "fir_no": "FIR-2026-TEST", "max_frames": 10, "stride": 1, "enhance_video": "false"}
        )
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["status"] == "SUCCESS"
    assert "exo_skeleton" in res_data["results"]
    assert "gait_kinematics" in res_data["results"]
    assert res_data["results"]["exo_skeleton"]["frames_extracted"] >= 0
    assert "model_training_dataset" in res_data["results"]
