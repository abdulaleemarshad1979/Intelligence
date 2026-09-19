"""Unit tests for perception model adapters (YOLO, OSNet, InsightFace, MMPose, GaitSet)."""

import numpy as np
import pytest
from app.adapters.detector_adapter import YOLOByteTrackAdapter
from app.adapters.reid_adapter import OSNetReIDAdapter, FastReIDAdapter
from app.adapters.face_adapter import InsightFaceArcFaceAdapter
from app.adapters.pose_adapter import MMPoseRTMPoseAdapter
from app.adapters.gait_adapter import GaitSetDynamicsAdapter

def test_detector_adapter():
    detector = YOLOByteTrackAdapter()
    info = detector.get_backend_info()
    assert "model_type" in info
    assert "backend" in info

    # Test detection on synthetic frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw simple bright rectangle simulating a person silhouette
    frame[100:300, 200:280] = 200

    results = detector.detect_and_track(frame, frame_id=1)
    assert isinstance(results, list)

def test_osnet_reid_adapter():
    reid = OSNetReIDAdapter(model_name="osnet_x1_0")
    crop = np.full((160, 64, 3), 128, dtype=np.uint8)
    crop[40:100, :, :] = 50  # Upper torso contrast

    res = reid.extract_embedding(crop)
    assert res.feature_dim == 512
    assert len(res.embedding) == 512
    # Verify L2 normalization
    norm = np.linalg.norm(res.embedding)
    assert 0.95 <= norm <= 1.05

def test_fastreid_adapter():
    fast_reid = FastReIDAdapter()
    crop = np.full((160, 64, 3), 100, dtype=np.uint8)
    res = fast_reid.extract_embedding(crop)
    assert res.feature_dim == 512
    assert len(res.embedding) == 512

def test_face_adapter_quality_and_tiers():
    face_adapter = InsightFaceArcFaceAdapter()
    crop = np.zeros((200, 100, 3), dtype=np.uint8)
    # Head region
    crop[10:50, 30:70] = [180, 200, 220]

    res = face_adapter.analyze_face(crop)
    assert hasattr(res, "quality_score")
    assert hasattr(res, "status")
    assert isinstance(res.tier_embeddings, dict)

def test_pose_adapter():
    pose_adapter = MMPoseRTMPoseAdapter()
    crop = np.zeros((200, 100, 3), dtype=np.uint8)
    crop[20:180, 30:70] = 200

    res = pose_adapter.estimate_pose(crop)
    assert hasattr(res, "keypoints")
    assert hasattr(res, "spine_tilt_deg")
    assert hasattr(res, "posture_score")
    assert 0.0 <= res.posture_score <= 1.0

def test_gait_adapter():
    gait_adapter = GaitSetDynamicsAdapter(sequence_length=16)
    track_hist = [
        {
            "frame_id": i,
            "inter_ankle_dist": 20.0 + 15.0 * np.sin(i * 0.4),
            "keypoints_crop": {
                "left_hip": [30, 80],
                "right_hip": [50, 80],
                "neck": [40, 30]
            }
        }
        for i in range(16)
    ]
    res = gait_adapter.analyze_sequence(track_hist)
    assert "stride_length_cm" in res
    assert "cadence_steps_per_sec" in res
    assert "gait_embedding" in res
    assert len(res["gait_embedding"]) == 32
