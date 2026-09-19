"""Tests for Calibrated Evidence Fusion Engine with Humility Vetoes."""

import pytest
from app.database.models import TrackObservation, CriminalRecord
from app.fusion.evidence import EvidenceFusionEngine


def test_height_contradiction_veto():
    engine = EvidenceFusionEngine(height_veto_threshold_cm=12.0)

    # Track of a tall person (188 cm) wearing similar clothes to a suspect
    track = TrackObservation(
        track_id="TRACK-TALL-01",
        camera_id="CAM-001",
        first_seen=0.0,
        last_seen=10.0,
        frame_count=25,
        face_visible=True,
        face_status="FULL_VISIBLE",
        estimated_height_cm=188.0,  # 188cm vs suspect 170cm -> delta 18cm (>12cm)
        body_proportions={"torso_leg_ratio": 0.85},
        clothing_upper="#1a2b3c",
        clothing_lower="#0f1011",
        stride_length_cm=75.0,
        spine_tilt_deg=2.0,
        posture_score=0.90,
        face_embedding=[0.2] * 128,
        body_embedding=[0.1] * 256,
        gait_embedding=[0.15] * 64
    )

    suspect = CriminalRecord(
        id="SUSP-SHORT-01",
        fir_no="FIR 110/2024",
        unit_name="East Godavari",
        subdivision="Central",
        police_station="I Town PS",
        accused_name="Test Suspect",
        known_height_cm=170.0,
        torso_leg_ratio=0.85,
        stride_length_cm=65.0,
        posture_lean_angle=2.0,
        posture_correctness=0.90,
        clothing_upper_color="#1a2b3c",
        clothing_lower_color="#0f1011",
        face_embedding=[0.2] * 128,
        body_embedding=[0.1] * 256,
        gait_embedding=[0.15] * 64
    )

    res = engine.evaluate_candidate(track, suspect)

    # Must be actively vetoed by humility rules
    assert res["is_vetoed"] is True
    assert res["humility_status"] == "VETO_TRIGGERED"
    assert any("Height disparity" in reason for reason in res["veto_reasons"])
    assert res["total_confidence"] <= 0.45
    assert res["status"] == "UNKNOWN_PERSON"
    assert "Humility Veto" in res["recommendation"]


def test_face_contradiction_veto():
    engine = EvidenceFusionEngine()

    # Track with different face embedding (orthogonal/opposite vector)
    track = TrackObservation(
        track_id="TRACK-FACE-DIFF",
        camera_id="CAM-001",
        first_seen=0.0,
        last_seen=10.0,
        frame_count=20,
        face_visible=True,
        face_status="FULL_VISIBLE",
        estimated_height_cm=172.0,
        body_proportions={"torso_leg_ratio": 0.85},
        clothing_upper="#1a2b3c",
        clothing_lower="#0f1011",
        stride_length_cm=65.0,
        spine_tilt_deg=2.0,
        posture_score=0.90,
        face_embedding=[-0.5] * 128,  # Contradicting face
        body_embedding=[0.1] * 256,
        gait_embedding=[0.15] * 64
    )

    suspect = CriminalRecord(
        id="SUSP-01",
        fir_no="FIR 110/2024",
        unit_name="East Godavari",
        subdivision="Central",
        police_station="I Town PS",
        accused_name="Test Suspect",
        known_height_cm=172.0,
        torso_leg_ratio=0.85,
        stride_length_cm=65.0,
        posture_lean_angle=2.0,
        posture_correctness=0.90,
        clothing_upper_color="#1a2b3c",
        clothing_lower_color="#0f1011",
        face_embedding=[0.5] * 128,
        body_embedding=[0.1] * 256,
        gait_embedding=[0.15] * 64
    )

    res = engine.evaluate_candidate(track, suspect)
    assert res["is_vetoed"] is True
    assert any("facial similarity" in r.lower() for r in res["veto_reasons"])
    assert res["status"] == "UNKNOWN_PERSON"


def test_concordant_evidence_achieves_high_confidence():
    engine = EvidenceFusionEngine()

    # All biometrics concordant
    track = TrackObservation(
        track_id="TRACK-MATCH",
        camera_id="CAM-001",
        first_seen=0.0,
        last_seen=10.0,
        frame_count=30,
        face_visible=True,
        face_status="FULL_VISIBLE",
        estimated_height_cm=173.0,
        body_proportions={"torso_leg_ratio": 0.85},
        clothing_upper="#5c4033",
        clothing_lower="#1f2421",
        stride_length_cm=65.0,
        spine_tilt_deg=4.0,
        posture_score=0.88,
        face_embedding=[0.3] * 128,
        body_embedding=[0.2] * 256,
        gait_embedding=[0.1] * 64
    )

    suspect = CriminalRecord(
        id="SUSP-MATCH",
        fir_no="FIR 184/2023",
        unit_name="East Godavari",
        subdivision="East Zone",
        police_station="II Town PS",
        accused_name="Korumilli Raju",
        known_height_cm=172.5,
        torso_leg_ratio=0.85,
        stride_length_cm=64.5,
        posture_lean_angle=4.2,
        posture_correctness=0.88,
        clothing_upper_color="#5c4033",
        clothing_lower_color="#1f2421",
        face_embedding=[0.3] * 128,
        body_embedding=[0.2] * 256,
        gait_embedding=[0.1] * 64
    )

    res = engine.evaluate_candidate(track, suspect)
    assert res["is_vetoed"] is False
    assert res["humility_status"] == "PASSED"
    assert res["total_confidence"] >= 0.78
    assert res["status"] == "HIGH_CONFIDENCE"
