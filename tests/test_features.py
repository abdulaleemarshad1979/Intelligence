"""Unit tests for multi-modal feature extraction modules."""

import numpy as np
import pytest
from app.features.face import FaceAnalyzer
from app.features.body import BodyAnalyzer
from app.features.height import HeightEstimator
from app.features.pose import PoseEstimator
from app.features.gait import GaitAnalyzer

def test_face_analyzer_tiers():
    analyzer = FaceAnalyzer()
    
    # Create synthetic test person crop (height 200, width 80)
    img = np.zeros((200, 80, 3), dtype=np.uint8)
    # Give head region some skin-like color (HSV ~ [15, 100, 150])
    img[0:50, 20:60] = [150, 180, 220]
    
    res = analyzer.analyze_person_crop(img)
    assert "tiers" in res
    assert "upper" in res["tiers"]
    assert "mid" in res["tiers"]
    assert "lower" in res["tiers"]
    assert len(res["face_embedding"]) == 128
    assert res["face_status"] in ["FULL_VISIBLE", "PARTIAL_UPPER", "MASKED_LOWER", "UNAVAILABLE"]

def test_body_analyzer():
    analyzer = BodyAnalyzer()
    img = np.zeros((220, 90, 3), dtype=np.uint8)
    # Upper shirt color (Brown)
    img[40:110, 20:70] = [50, 60, 90]
    # Lower trousers (Navy)
    img[120:200, 20:70] = [60, 40, 20]

    res = analyzer.analyze(img)
    assert res["torso_length_px"] > 0
    assert res["leg_length_px"] > 0
    assert "torso_leg_ratio" in res["proportions"]
    assert res["clothing"]["upper_hex"].startswith("#")
    assert res["clothing"]["lower_hex"].startswith("#")
    assert len(res["body_embedding"]) == 256

def test_height_estimator():
    estimator = HeightEstimator(mounting_height_m=4.2, tilt_deg=35.0, focal_length_px=950.0)
    # Bounding box near bottom of frame
    box = [400, 150, 520, 480]  # height = 330px
    res = estimator.estimate_height_cm(box, frame_height=576)
    
    assert 150.0 <= res["estimated_height_cm"] <= 195.0
    assert res["raw_pixel_height"] == 330
    assert res["confidence"] > 0.5

def test_gait_analyzer():
    analyzer = GaitAnalyzer(window_size=20)
    # Simulate oscillating inter-ankle distance sequence
    pose_history = []
    for i in range(25):
        stride = 20.0 + 15.0 * np.sin(i * 0.5)
        pose_history.append({
            "keypoints_crop": {
                "neck": [40, 30],
                "left_hip": [30, 100],
                "right_hip": [50, 100],
                "left_ankle": [40 - stride/2, 190],
                "right_ankle": [40 + stride/2, 190],
            },
            "inter_ankle_dist": stride
        })

    res = analyzer.analyze_sequence(pose_history, estimated_height_cm=172.0)
    assert res["stride_length_cm"] > 0.0
    assert res["cadence_steps_per_sec"] > 0.0
    assert len(res["gait_wave"]) > 0
    assert len(res["gait_embedding"]) == 64
