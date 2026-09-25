"""Tests for Section 3: Signal Fusion Architecture & Disparity Veto.

Validates the core principle:
"At 1:N scale across an entire city or district gallery, soft biometrics alone yield
unacceptably high false-match rates. Soft signals must act as conditional confirmations
or hard geometric pruning gates rather than independent identity verifiers."
"""

import pytest
from app.database.models import TrackObservation, CriminalRecord
from app.fusion.disparity_veto import DisparityVetoGate
from app.fusion.evidence import EvidenceFusionEngine
from app.reid.matcher import CandidateMatcher
from app.database.repository import Repository


def test_geometric_pruning_gate_height():
    gate = DisparityVetoGate(max_height_disparity_cm=12.0)
    # 188cm track vs 170cm suspect -> delta 18cm > 12cm
    is_pruned, reasons = gate.evaluate_geometric_pruning(
        track_height_cm=188.0,
        suspect_height_cm=170.0,
        track_ratio=0.85,
        suspect_ratio=0.85,
        track_stride_cm=65.0,
        suspect_stride_cm=65.0
    )
    assert is_pruned is True
    assert any("Height disparity" in r for r in reasons)


def test_geometric_pruning_gate_proportions():
    gate = DisparityVetoGate(max_ratio_disparity=0.30)
    # Ratio 0.60 vs 0.95 -> delta 0.35 > 0.30
    is_pruned, reasons = gate.evaluate_geometric_pruning(
        track_height_cm=172.0,
        suspect_height_cm=172.0,
        track_ratio=0.60,
        suspect_ratio=0.95,
        track_stride_cm=65.0,
        suspect_stride_cm=65.0
    )
    assert is_pruned is True
    assert any("proportion disparity" in r.lower() for r in reasons)


def test_geometric_pruning_gate_kinematic_stride():
    gate = DisparityVetoGate(max_stride_disparity_cm=25.0)
    # Stride 95cm vs 65cm -> delta 30cm > 25cm
    is_pruned, reasons = gate.evaluate_geometric_pruning(
        track_height_cm=172.0,
        suspect_height_cm=172.0,
        track_ratio=0.85,
        suspect_ratio=0.85,
        track_stride_cm=95.0,
        suspect_stride_cm=65.0
    )
    assert is_pruned is True
    assert any("stride kinematics disparity" in r.lower() for r in reasons)


def test_soft_biometrics_cannot_independently_verify_at_1_to_n():
    """When face is unverified or rear-view, soft biometrics alone cannot produce HIGH_CONFIDENCE."""
    engine = EvidenceFusionEngine(enforce_1_to_n_soft_cap=True)


    # Track with identical clothing, gait, and height, but NO face (rear view)
    track = TrackObservation(
        track_id="TRACK-SOFT-ONLY",
        camera_id="CAM-001",
        first_seen=0.0,
        last_seen=10.0,
        frame_count=30,
        face_visible=False,
        face_status="UNAVAILABLE",
        estimated_height_cm=172.0,
        body_proportions={"torso_leg_ratio": 0.85},
        clothing_upper="#1a2b3c",
        clothing_lower="#0f1011",
        stride_length_cm=65.0,
        spine_tilt_deg=2.0,
        posture_score=0.95,
        face_embedding=[],
        body_embedding=[0.5] * 256,
        gait_embedding=[0.5] * 64
    )

    suspect = CriminalRecord(
        id="SUSP-MATCH-SOFT",
        fir_no="FIR 200/2024",
        unit_name="East Godavari",
        subdivision="Central",
        police_station="I Town PS",
        accused_name="Gait Test Suspect",
        known_height_cm=172.0,
        torso_leg_ratio=0.85,
        stride_length_cm=65.0,
        posture_lean_angle=2.0,
        posture_correctness=0.95,
        clothing_upper_color="#1a2b3c",
        clothing_lower_color="#0f1011",
        face_embedding=[0.8] * 128,
        body_embedding=[0.5] * 256,
        gait_embedding=[0.5] * 64
    )

    res = engine.evaluate_candidate(track, suspect)

    # Core Principle check: Soft biometrics CANNOT verify identity alone at 1:N scale
    assert res["status"] != "HIGH_CONFIDENCE"
    assert res["total_confidence"] <= 0.45
    assert "Conditional Confirmation Only" in res["recommendation"]
    assert "signal_fusion_rule" in res


def test_primary_face_with_soft_conditional_confirmation():
    """When primary face is available, soft biometrics conditionally confirm and boost."""
    engine = EvidenceFusionEngine()

    track = TrackObservation(
        track_id="TRACK-PRIMARY-CONFIRMED",
        camera_id="CAM-001",
        first_seen=0.0,
        last_seen=10.0,
        frame_count=30,
        face_visible=True,
        face_status="FULL_VISIBLE",
        estimated_height_cm=172.0,
        body_proportions={"torso_leg_ratio": 0.85},
        clothing_upper="#1a2b3c",
        clothing_lower="#0f1011",
        stride_length_cm=65.0,
        spine_tilt_deg=2.0,
        posture_score=0.95,
        face_embedding=[0.8] * 128,
        body_embedding=[0.5] * 256,
        gait_embedding=[0.5] * 64
    )

    suspect = CriminalRecord(
        id="SUSP-CONFIRMED",
        fir_no="FIR 201/2024",
        unit_name="East Godavari",
        subdivision="Central",
        police_station="I Town PS",
        accused_name="Confirmed Suspect",
        known_height_cm=172.0,
        torso_leg_ratio=0.85,
        stride_length_cm=65.0,
        posture_lean_angle=2.0,
        posture_correctness=0.95,
        clothing_upper_color="#1a2b3c",
        clothing_lower_color="#0f1011",
        face_embedding=[0.8] * 128,
        body_embedding=[0.5] * 256,
        gait_embedding=[0.5] * 64
    )

    res = engine.evaluate_candidate(track, suspect)
    assert res["is_vetoed"] is False
    assert res["status"] == "HIGH_CONFIDENCE"
    assert res["total_confidence"] >= 0.78
