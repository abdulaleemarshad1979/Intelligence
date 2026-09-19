"""Unit tests for Evidence Fusion Engine."""

import pytest
from app.database.models import TrackObservation, CriminalRecord
from app.fusion.evidence import EvidenceFusionEngine

def test_evidence_fusion_face_visible():
    engine = EvidenceFusionEngine()

    track = TrackObservation(
        track_id="TRACK-TEST-01",
        camera_id="CAM-001",
        first_seen=0.0,
        last_seen=10.0,
        frame_count=20,
        face_visible=True,
        face_status="FULL_VISIBLE",
        estimated_height_cm=173.0,
        body_proportions={"torso_leg_ratio": 0.85},
        clothing_upper="#5c4033",
        clothing_lower="#1f2421",
        stride_length_cm=65.0,
        spine_tilt_deg=4.0,
        posture_score=0.85,
        face_embedding=[0.1] * 128,
        body_embedding=[0.05] * 256,
        gait_embedding=[0.1] * 64
    )

    suspect = CriminalRecord(
        id="SUSP-01",
        fir_no="FIR 184/2023",
        unit_name="East Godavari",
        subdivision="East Zone",
        police_station="II Town PS",
        accused_name="Korumilli Raju",
        known_height_cm=172.5,
        torso_leg_ratio=0.86,
        stride_length_cm=64.0,
        posture_lean_angle=5.0,
        posture_correctness=0.82,
        clothing_upper_color="#5c4033",
        clothing_lower_color="#1f2421",
        face_embedding=[0.1] * 128,
        body_embedding=[0.05] * 256,
        gait_embedding=[0.1] * 64
    )

    res = engine.evaluate_candidate(track, suspect)
    assert res["is_face_available"] is True
    assert res["weights_used"]["face"] == 0.40
    assert res["total_confidence"] > 0.70
    assert res["status"] in ["HIGH_CONFIDENCE", "REVIEW_REQUIRED"]

def test_evidence_fusion_face_unavailable_masked():
    engine = EvidenceFusionEngine()

    track = TrackObservation(
        track_id="TRACK-TEST-02",
        camera_id="CAM-001",
        first_seen=0.0,
        last_seen=10.0,
        frame_count=20,
        face_visible=False,
        face_status="UNAVAILABLE",  # Rear view or surgical mask
        estimated_height_cm=173.0,
        body_proportions={"torso_leg_ratio": 0.85},
        clothing_upper="#5c4033",
        clothing_lower="#1f2421",
        stride_length_cm=65.0,
        spine_tilt_deg=4.0,
        posture_score=0.85,
        face_embedding=[0.0] * 128,
        body_embedding=[0.05] * 256,
        gait_embedding=[0.1] * 64
    )

    suspect = CriminalRecord(
        id="SUSP-01",
        fir_no="FIR 184/2023",
        unit_name="East Godavari",
        subdivision="East Zone",
        police_station="II Town PS",
        accused_name="Korumilli Raju",
        known_height_cm=172.5,
        torso_leg_ratio=0.86,
        stride_length_cm=64.0,
        posture_lean_angle=5.0,
        posture_correctness=0.82,
        clothing_upper_color="#5c4033",
        clothing_lower_color="#1f2421",
        face_embedding=[0.1] * 128,
        body_embedding=[0.05] * 256,
        gait_embedding=[0.1] * 64
    )

    res = engine.evaluate_candidate(track, suspect)
    assert res["is_face_available"] is False
    # When face is unavailable, face weight must be 0.00 and body/gait/height absorb the weights!
    assert res["weights_used"]["face"] == 0.00
    assert res["weights_used"]["body"] == 0.40
    assert res["weights_used"]["gait"] == 0.45
    assert res["weights_used"]["height"] == 0.15
    assert res["scores"]["face_score"] == 0.0
    assert res["total_confidence"] > 0.70  # Still high confidence candidate based on body, gait, height!
    assert res["face_evidence_status"] == "UNAVAILABLE (Masked/Rear/Blur)"
