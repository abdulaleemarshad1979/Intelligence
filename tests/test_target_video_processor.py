"""Integration and unit tests for Video Target Processor, Exo-Skeleton Extraction, and SQL Training Export."""

import os
import json
import tempfile
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.pipeline.video_target_processor import VideoTargetProcessor
from app.database.database import init_db
from app.database.repository import Repository
from app.main import app


@pytest.fixture
def temp_workspace(tmp_path):
    db_file = str(tmp_path / "test_records.db")
    init_db(db_file)
    repo = Repository(db_file)
    output_dir = str(tmp_path / "captures")
    os.makedirs(output_dir, exist_ok=True)
    return {
        "db_file": db_file,
        "repo": repo,
        "output_dir": output_dir,
        "tmp_path": tmp_path
    }


def create_synthetic_walking_video(video_path: str, num_frames: int = 25, w: int = 320, h: int = 240):
    """Creates a synthetic video of a moving figure."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 25.0, (w, h))

    for i in range(num_frames):
        frame = np.full((h, w, 3), 40, dtype=np.uint8)
        # Person moving horizontally
        px = 40 + i * 8
        py = 50
        pw, ph = 60, 140

        # Draw pedestrian figure
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (180, 180, 180), -1)
        # Head
        cv2.circle(frame, (px + pw // 2, py + 20), 15, (200, 190, 180), -1)
        # Torso
        cv2.rectangle(frame, (px + 10, py + 35), (px + pw - 10, py + 85), (60, 80, 140), -1)
        # Legs
        stride = 10 * np.sin(i * 0.5)
        cv2.line(frame, (px + 20, py + 85), (int(px + 15 - stride), py + ph), (30, 40, 60), 6)
        cv2.line(frame, (px + 40, py + 85), (int(px + 45 + stride), py + ph), (30, 40, 60), 6)

        out.write(frame)

    out.release()


def test_video_target_processor_end_to_end(temp_workspace):
    video_path = str(temp_workspace["tmp_path"] / "test_walk.mp4")
    create_synthetic_walking_video(video_path, num_frames=20)

    processor = VideoTargetProcessor(
        db_path=temp_workspace["db_file"],
        conf_threshold=0.25,
        enable_clahe=True,
        enable_denoise=True
    )

    results = processor.process_video_target(
        video_path=video_path,
        target_name="Test Suspect",
        target_id="POI-TEST-99",
        camera_id="CAM-TEST-01",
        max_frames=20,
        stride_step=1,
        enhance_video=True,
        output_dir=temp_workspace["output_dir"]
    )

    assert results["status"] == "SUCCESS"
    assert results["target"]["target_id"] == "POI-TEST-99"
    assert results["target"]["name"] == "Test Suspect"

    # Verify SQL persistence
    repo = temp_workspace["repo"]
    skeletons = repo.get_skeletons_for_track(results["target"]["track_id"])
    assert len(skeletons) > 0
    first_skel = skeletons[0]
    assert "keypoints_coco_17" in first_skel
    assert "inter_ankle_dist" in first_skel
    assert "confidences" in first_skel
    assert first_skel["person_id"] == "POI-TEST-99"

    # Verify gait dynamics in SQL
    gait = repo.get_gait_dynamics(results["target"]["track_id"])
    assert gait is not None
    assert "stride_length_cm" in gait
    assert "cadence_hz" in gait

    # Verify model training dataset export
    train_info = results["model_training_dataset"]
    assert os.path.isfile(train_info["npy_path"])
    assert os.path.isfile(train_info["metadata_path"])
    npy_data = np.load(train_info["npy_path"])
    assert len(npy_data.shape) == 3
    assert npy_data.shape[1] == 17  # 17 COCO joints
    assert npy_data.shape[2] == 3   # (x, y, confidence)

    # Verify video enhancement artifact
    enh_info = results["video_enhancement"]
    assert enh_info["enhanced"] is True
    assert os.path.isfile(enh_info["enhanced_video_path"])
    assert len(enh_info["raw_sha256"]) == 64
    assert len(enh_info["enhanced_sha256"]) == 64


def test_api_video_target_endpoints(temp_workspace):
    client = TestClient(app)

    video_path = str(temp_workspace["tmp_path"] / "api_test_walk.mp4")
    create_synthetic_walking_video(video_path, num_frames=15)

    resp = client.post(
        "/api/video/process-target",
        json={
            "video_path": video_path,
            "target_name": "API Subject",
            "target_id": "POI-API-01",
            "max_frames": 15,
            "stride": 1,
            "enhance_video": False,
            "output_dir": temp_workspace["output_dir"]
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["target"]["target_id"] == "POI-API-01"

    # Test training export endpoint
    export_resp = client.get("/api/training/export")
    assert export_resp.status_code == 200
    export_data = export_resp.json()
    assert export_data["status"] == "SUCCESS"
    assert "dataset" in export_data
