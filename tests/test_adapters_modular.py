"""Unit tests for individual modular model adapters across all categories."""

import numpy as np
import pytest
from app.adapters.detection.yolo import YOLODetectorAdapter
from app.adapters.detection.rtdetr import RTDETRDetectorAdapter
from app.adapters.tracking.bytetrack import ByteTrackAdapter
from app.adapters.tracking.botsort import BoTSORTAdapter
from app.adapters.tracking.deepstream import DeepStreamTrackerAdapter
from app.adapters.tracking.mmtracking import MMTrackingAdapter
from app.adapters.reid.osnet import OSNetReIDAdapter
from app.adapters.reid.fastreid import FastReIDAdapter
from app.adapters.face.insightface import InsightFaceArcFaceAdapter
from app.adapters.pose.rtmpose import RTMPoseAdapter
from app.adapters.pose.mmpose import MMPoseAdapter
from app.adapters.gait.opengait import OpenGaitAdapter
from app.adapters.gait.gaitset import GaitSetAdapter
from app.adapters.base import DetectionResult


def test_modular_detectors():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:300, 200:280] = 200

    yolo = YOLODetectorAdapter()
    yolo_dets = yolo.detect_and_track(frame, 1)
    assert isinstance(yolo_dets, list)
    assert yolo.get_backend_info()["model_id"] == "yolo"

    rtdetr = RTDETRDetectorAdapter()
    rt_dets = rtdetr.detect_and_track(frame, 1)
    assert isinstance(rt_dets, list)
    assert rtdetr.get_backend_info()["model_id"] == "rtdetr"


def test_modular_trackers():
    detections = [
        DetectionResult(bbox=(100, 100, 50, 120), confidence=0.85, track_id=None),
        DetectionResult(bbox=(300, 200, 60, 150), confidence=0.90, track_id=None)
    ]

    # ByteTrack
    bt = ByteTrackAdapter()
    bt_res = bt.update(detections, frame_id=1)
    assert isinstance(bt_res, list)

    # BoT-SORT
    bs = BoTSORTAdapter()
    bs_res = bs.update(detections, frame_id=1)
    assert isinstance(bs_res, list)

    # DeepStream
    ds = DeepStreamTrackerAdapter()
    ds_res = ds.update(detections, frame_id=1)
    assert isinstance(ds_res, list)

    # MMTracking
    mm = MMTrackingAdapter()
    mm_res = mm.update(detections, frame_id=1)
    assert isinstance(mm_res, list)


def test_modular_reid():
    crop = np.full((160, 64, 3), 120, dtype=np.uint8)
    crop[40:100, :, :] = 60

    osnet = OSNetReIDAdapter()
    os_res = osnet.extract_embedding(crop)
    assert os_res.feature_dim == 512
    assert len(os_res.embedding) == 512
    assert 0.95 <= np.linalg.norm(os_res.embedding) <= 1.05

    fastreid = FastReIDAdapter()
    fr_res = fastreid.extract_embedding(crop)
    assert fr_res.feature_dim == 512
    assert len(fr_res.embedding) == 512


def test_modular_face():
    face_adapter = InsightFaceArcFaceAdapter()
    crop = np.zeros((200, 100, 3), dtype=np.uint8)
    crop[10:50, 30:70] = [180, 200, 220]

    res = face_adapter.analyze_face(crop)
    assert hasattr(res, "quality_score")
    assert hasattr(res, "status")
    assert "license_warning" in face_adapter.get_backend_info()


def test_modular_pose():
    crop = np.zeros((200, 100, 3), dtype=np.uint8)
    crop[20:180, 30:70] = 200

    rtmpose = RTMPoseAdapter()
    rt_res = rtmpose.estimate_pose(crop)
    assert len(rt_res.keypoints) == 17
    assert 0.0 <= rt_res.posture_score <= 1.0

    mmpose = MMPoseAdapter()
    mm_res = mmpose.estimate_pose(crop)
    assert len(mm_res.keypoints) == 17


def test_opengait_contract():
    """Verify OpenGaitAdapter implements extract_sequence, generate_embedding, quality_score."""
    opengait = OpenGaitAdapter(model_name="GaitBase", sequence_length=16, feature_dim=64)

    track_hist = [
        {
            "frame_id": i,
            "inter_ankle_dist": 20.0 + 10.0 * np.sin(i * 0.5),
            "keypoints_crop": {
                "left_ankle": [30.0 + 5.0 * np.cos(i * 0.5), 80.0],
                "right_ankle": [50.0 - 5.0 * np.cos(i * 0.5), 80.0]
            }
        }
        for i in range(20)
    ]

    # 1. extract_sequence
    seq = opengait.extract_sequence(track_hist)
    assert len(seq) <= 16

    # 2. quality_score
    q = opengait.quality_score(seq)
    assert 0.0 <= q <= 1.0

    # 3. generate_embedding
    emb = opengait.generate_embedding(seq, estimated_height_cm=175.0)
    assert len(emb) == 64
    norm = np.linalg.norm(emb)
    assert 0.95 <= norm <= 1.05

    # 4. analyze_sequence
    res = opengait.analyze_sequence(track_hist, estimated_height_cm=175.0)
    assert "gait_embedding" in res
    assert len(res["gait_embedding"]) == 64
    assert "quality_score" in res
    assert "license_warning" in opengait.get_backend_info()


def test_gaitset_adapter():
    gaitset = GaitSetAdapter(sequence_length=16)
    track_hist = [{"frame_id": i, "inter_ankle_dist": 25.0} for i in range(16)]
    res = gaitset.analyze_sequence(track_hist)
    assert len(res["gait_embedding"]) == 32
    assert res["quality_score"] == 1.0
