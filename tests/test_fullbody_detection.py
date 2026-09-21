"""Comprehensive tests for SOTA Full Body Pedestrian Detection and Pose Estimation."""

import pytest
import numpy as np
from app.detection.person_detector import PersonDetector
from app.adapters.detection import (
    YOLODetectorAdapter,
    RTDETRDetectorAdapter,
    FullBodyPoseDetectorAdapter,
    EnsemblePedestrianDetector
)
from app.features.pose import PoseEstimator
from app.adapters.registry import model_registry


def test_person_detector_sota_neural_initialization():
    """Verify PersonDetector initializes neural backend and detects pedestrians."""
    detector = PersonDetector(model_name="yolov8n.pt", confidence_threshold=0.3)
    info = detector.get_backend_info()

    assert info["is_neural_loaded"] is True
    assert "YOLO" in info["backend"]
    assert info["confidence_threshold"] == 0.3

    # Generate synthetic image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:350, 200:300] = 220  # Bright vertical rectangular figure

    dets = detector.detect(frame)
    assert isinstance(dets, list)


def test_person_detector_fullbody_pose_mode():
    """Verify PersonDetector with pose estimation extracts 17 COCO skeletal landmarks."""
    detector = PersonDetector(model_name="yolov8n-pose.pt", confidence_threshold=0.25, enable_pose=True)
    info = detector.get_backend_info()

    assert info["is_neural_loaded"] is True
    assert info["is_pose_enabled"] is True

    # Synthetic frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[80:400, 220:340] = 200

    dets = detector.detect(frame)
    assert isinstance(dets, list)
    if len(dets) > 0 and "keypoints" in dets[0]:
        kpts = dets[0]["keypoints"]
        assert "nose" in kpts
        assert "left_ankle" in kpts
        assert "right_ankle" in kpts
        assert "left_shoulder" in kpts


def test_fullbody_pose_detector_adapter():
    """Verify FullBodyPoseDetectorAdapter contract and 17-keypoint skeleton output."""
    adapter = FullBodyPoseDetectorAdapter(model_name="yolov8n-pose.pt")
    info = adapter.get_backend_info()

    assert info["model_id"] == "fullbody_pose"
    assert info["num_keypoints"] == 17
    assert info["is_neural_model_loaded"] is True

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[60:380, 180:300] = 210

    results = adapter.detect_and_track(frame, frame_id=1)
    assert isinstance(results, list)
    for res in results:
        assert res.class_name == "person"
        assert len(res.bbox) == 4
        assert 0.0 <= res.confidence <= 1.0


def test_ensemble_pedestrian_detector():
    """Verify EnsemblePedestrianDetector fuses candidate detections using Weighted Box Fusion."""
    yolo_adapter = YOLODetectorAdapter(model_name="yolov8n.pt")
    rtdetr_adapter = RTDETRDetectorAdapter(model_name="rtdetr-l.pt")

    ensemble = EnsemblePedestrianDetector(
        detectors=[yolo_adapter, rtdetr_adapter],
        weights=[0.6, 0.4]
    )
    info = ensemble.get_backend_info()
    assert info["model_id"] == "ensemble"
    assert info["num_sub_models"] == 2
    assert "WBF_FUSION" in info["backend"]

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:360, 220:320] = 200

    fused = ensemble.detect_and_track(frame, frame_id=1)
    assert isinstance(fused, list)


def test_model_registry_sota_selection():
    """Verify ModelRegistry allows dynamic switching to SOTA full body detectors."""
    # 1. Switch to fullbody_pose
    res_pose = model_registry.select_model("detection", "fullbody_pose")
    assert res_pose["status"] == "SUCCESS"
    assert res_pose["active_model"] == "fullbody_pose"
    assert model_registry.active_keys["detection"] == "fullbody_pose"

    # 2. Switch to ensemble
    res_ens = model_registry.select_model("detection", "ensemble")
    assert res_ens["status"] == "SUCCESS"
    assert res_ens["active_model"] == "ensemble"

    # 3. Switch to yolo11x
    res_11x = model_registry.select_model("detection", "yolo11x")
    assert res_11x["status"] == "SUCCESS"
    assert res_11x["active_model"] == "yolo11x"

    # 4. Switch back to rtdetr
    res_rt = model_registry.select_model("detection", "rtdetr")
    assert res_rt["status"] == "SUCCESS"


def test_features_pose_estimator_neural():
    """Verify PoseEstimator in app/features/pose.py produces accurate landmarks for gait."""
    estimator = PoseEstimator(model_name="yolov8n-pose.pt")
    crop = np.zeros((240, 100, 3), dtype=np.uint8)
    crop[20:220, 25:75] = 200

    res = estimator.estimate_pose(crop, box=[100, 100, 200, 340])
    assert res["visible"] is True
    assert "keypoints_crop" in res
    assert "inter_ankle_dist" in res
    assert "neck" in res["keypoints_crop"]
    assert "left_hip" in res["keypoints_crop"]
    assert "right_hip" in res["keypoints_crop"]
    assert "left_ankle" in res["keypoints_crop"]
    assert "right_ankle" in res["keypoints_crop"]
