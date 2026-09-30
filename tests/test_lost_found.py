"""Unit and integration tests for Pushkaralu Lost & Found module."""

import io
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.missing_person.lost_found import (
    LFConfig,
    FaceAnalyzer,
    MediaPipePose,
    RTMPoseSource,
    LostFoundSearch,
    CaseMeta,
    QueryProfile,
    enroll_case,
    body_ratio,
    gait_features,
    gait_similarity,
    colour_fraction,
    clothing_regions,
    build_router,
    build_calibrated_height_fn,
    load_cam_xy,
    J
)


@pytest.fixture
def lf_config():
    return LFConfig(
        sface_cosine_threshold=0.35,
        min_track_frames=2,
        face_every_n=1,
        pose_every_n=1,
        review_threshold=0.40,
        high_conf_threshold=0.70
    )


@pytest.fixture
def mock_pose_source():
    class DummyPoseSource:
        def keypoints(self, img, bbox):
            out = np.zeros((14, 3), np.float32)
            # Fill with realistic normalized joint positions
            out[J["head"]] = (100, 40, 0.9)
            out[J["neck"]] = (100, 60, 0.9)
            out[J["r_sho"]] = (80, 60, 0.9)
            out[J["l_sho"]] = (120, 60, 0.9)
            out[J["r_hip"]] = (85, 120, 0.9)
            out[J["l_hip"]] = (115, 120, 0.9)
            out[J["r_knee"]] = (85, 170, 0.9)
            out[J["l_knee"]] = (115, 170, 0.9)
            out[J["r_ank"]] = (85, 220, 0.9)
            out[J["l_ank"]] = (115, 220, 0.9)
            return out
    return DummyPoseSource()


def test_body_ratio(mock_pose_source):
    kp = mock_pose_source.keypoints(np.zeros((300, 200, 3), np.uint8), (0, 0, 200, 300))
    ratio = body_ratio(kp)
    assert ratio is not None
    assert 0.4 <= ratio <= 1.0


def test_gait_features():
    # Synthesize 50 walking frames at 25 fps
    fps = 25.0
    kps = []
    for f in range(60):
        kp = np.zeros((14, 3), np.float32)
        kp[:, 2] = 0.9  # high confidence
        # Neck & hip
        kp[J["neck"]] = (100, 50, 0.9)
        kp[J["r_hip"]] = (90, 100 + 2.0 * np.sin(2 * np.pi * 1.5 * (f / fps)), 0.9)
        kp[J["l_hip"]] = (110, 100 + 2.0 * np.sin(2 * np.pi * 1.5 * (f / fps)), 0.9)
        # Ankle stride oscillation
        stride = 20.0 * np.sin(2 * np.pi * 1.5 * (f / fps))
        kp[J["r_ank"]] = (100 + stride, 200, 0.9)
        kp[J["l_ank"]] = (100 - stride, 200, 0.9)
        kps.append(kp)

    feat = gait_features(kps, fps)
    assert feat is not None
    assert len(feat) == 5
    sim = gait_similarity(feat, feat)
    assert sim > 0.95


def test_clothing_analysis():
    # Create an image with red upper and blue lower
    img = np.zeros((200, 100, 3), dtype=np.uint8)
    # Red in BGR is (0, 0, 255)
    img[20:100, :] = (0, 0, 255)
    # Blue in BGR is (255, 0, 0)
    img[100:180, :] = (255, 0, 0)

    bbox = (0, 0, 100, 200)
    up, lo = clothing_regions(img, None, bbox)
    assert up.size > 0 and lo.size > 0

    red_frac = colour_fraction(up, "red")
    assert red_frac > 0.6
    blue_frac = colour_fraction(lo, "blue")
    assert blue_frac > 0.6


def test_lost_found_lifecycle(lf_config, mock_pose_source):
    face_analyzer = FaceAnalyzer(lf_config)
    search_engine = LostFoundSearch(
        cfg=lf_config,
        face=face_analyzer,
        pose=mock_pose_source,
        cam_positions_m={"CAM-001": (0.0, 0.0), "CAM-002": (50.0, 10.0)}
    )

    meta = CaseMeta(
        officer_id="OFFICER-77",
        guardian_contact="+91-9876543210",
        display_name="Missing Person Test",
        age=30,
        height_cm=165.0,
        upper_colours=["red"],
        lower_colours=["blue"],
        last_seen_cam="CAM-001",
        last_seen_ts=100.0
    )

    # Synthetic reference photo
    photo = np.zeros((200, 200, 3), dtype=np.uint8)
    photo[40:120, :] = (0, 0, 255)  # upper red
    photo[120:190, :] = (255, 0, 0) # lower blue

    profile = enroll_case([photo], None, meta, face_analyzer, mock_pose_source, lf_config)
    assert profile.case_id.startswith("LF-")
    search_engine.open_case(profile)
    assert profile.case_id in search_engine.cases

    # Observe a live person matching clothing and body ratio
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    frame[50:150, 50:150] = (0, 0, 255)
    frame[150:250, 50:150] = (255, 0, 0)

    for i in range(10):
        search_engine.observe(
            cam_id="CAM-001",
            track_id=101,
            frame=frame,
            bbox=(50.0, 30.0, 150.0, 270.0),
            ts=110.0 + i,
            fps=25.0
        )

    # Check alert queue
    alerts = search_engine.pop_alerts()
    # If no face, soft cues (clothing/body) will generate REVIEW_NO_FACE if raw score passes
    # Verify candidate properties if generated
    for a in alerts:
        assert a.case_id == profile.case_id
        assert a.tier in ("HIGH_CONFIDENCE", "REVIEW", "REVIEW_NO_FACE")

    # Close case and check biometric purging
    search_engine.close_case(profile.case_id, "FOUND", "OFFICER-77")
    assert profile.case_id not in search_engine.cases
    assert len(profile.face_embs) == 0
    assert profile.face_geom is None


def test_api_routes(lf_config, mock_pose_source):
    from fastapi import FastAPI
    face_analyzer = FaceAnalyzer(lf_config)
    search_engine = LostFoundSearch(
        cfg=lf_config,
        face=face_analyzer,
        pose=mock_pose_source
    )

    app = FastAPI()
    app.include_router(build_router(search_engine))
    client = TestClient(app)

    # Create dummy JPEG image
    import cv2
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", dummy_img)
    img_bytes = buf.tobytes()

    response = client.post(
        "/api/missing/cases",
        data={
            "officer_id": "AP-POLICE-01",
            "guardian_contact": "+91-9988776655",
            "display_name": "Ramesh Kumar",
            "age": "45",
            "upper_colours": "white, saffron",
            "lower_colours": "black"
        },
        files=[("photos", ("test.jpg", img_bytes, "image/jpeg"))]
    )
    assert response.status_code == 200
    res_data = response.json()
    assert "case_id" in res_data
    case_id = res_data["case_id"]

    # List cases
    list_res = client.get("/api/missing/cases")
    assert list_res.status_code == 200
    assert any(c["case_id"] == case_id for c in list_res.json())

    # Get alerts
    alerts_res = client.get("/api/missing/alerts")
    assert alerts_res.status_code == 200

    # Close case
    close_res = client.post(f"/api/missing/cases/{case_id}/close", data={"status": "FOUND", "officer_id": "AP-POLICE-01"})
    assert close_res.status_code == 200
    assert close_res.json()["status"] == "FOUND"
