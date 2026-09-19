"""Unit tests for Spatio-Temporal Retrograde Trajectory Reconstruction.

Tests:
- Kinematic velocity bounds validation (0.5 m/s <= v <= 3.0 m/s)
- Pruning of physically impossible visual matches across the urban street network
- Backward-in-time multi-hop graph traversal to isolate suspect origin loci
- Loop prevention and terminal condition handling
"""

import pytest
import numpy as np
from app.graph.retrograde import RetrogradeTracker


@pytest.fixture
def sample_distances():
    # CAM-001 <--- 150m ---> CAM-002 <--- 120m ---> CAM-003
    return {
        "CAM-001": {"CAM-002": 150.0},
        "CAM-002": {"CAM-001": 150.0, "CAM-003": 120.0},
        "CAM-003": {"CAM-002": 120.0}
    }


def test_kinematic_feasibility_bounds(sample_distances):
    tracker = RetrogradeTracker(camera_distances=sample_distances, v_min=0.5, v_max=3.0)

    # 150 meters between CAM-001 and CAM-002:
    # min_time = 150 / 3.0 = 50.0s
    # max_time = 150 / 0.5 = 300.0s

    # Normal walking pace (1.5 m/s -> 100s) -> FEASIBLE
    assert tracker.is_kinematically_feasible("CAM-001", "CAM-002", delta_t=100.0) is True

    # Sprint pace at boundary (3.0 m/s -> 50.0s) -> FEASIBLE
    assert tracker.is_kinematically_feasible("CAM-001", "CAM-002", delta_t=50.0) is True

    # Physically impossible sprint (15.0 m/s -> 10.0s) -> INFEASIBLE (Pruned)
    assert tracker.is_kinematically_feasible("CAM-001", "CAM-002", delta_t=10.0) is False

    # Stalled beyond max transit window (0.3 m/s -> 500.0s) -> INFEASIBLE
    assert tracker.is_kinematically_feasible("CAM-001", "CAM-002", delta_t=500.0) is False

    # Negative time or reverse causality -> INFEASIBLE
    assert tracker.is_kinematically_feasible("CAM-001", "CAM-002", delta_t=-10.0) is False

    # Same camera -> FEASIBLE for non-negative delta_t
    assert tracker.is_kinematically_feasible("CAM-001", "CAM-001", delta_t=5.0) is True


def test_retrograde_backward_origin_reconstruction(sample_distances):
    tracker = RetrogradeTracker(camera_distances=sample_distances, v_min=0.5, v_max=3.0, score_threshold=0.70)

    t0 = 1000.0
    # Standard normalized 512-dim embedding for target
    emb_target = [0.0] * 512
    emb_target[0] = 0.6
    emb_target[1] = 0.8  # norm = 1.0

    # Probe at crime scene (CAM-002) at t0
    incident_probe = {
        "track_id": "PROBE-CRIME-SCENE",
        "camera_id": "CAM-002",
        "t_in": t0,
        "t_out": t0 + 20.0,
        "embedding": emb_target,
        "active_attributes": ["backpack", "upper_black", "lower_blue"]
    }

    # Candidate pool in history
    # Candidate 1: Genuine origin at CAM-001 (arrived CAM-002 in 100s, matching emb & attr)
    cand_genuine = {
        "track_id": "TRK-ORIGIN-CAM001",
        "camera_id": "CAM-001",
        "t_in": t0 - 150.0,
        "t_out": t0 - 100.0,  # delta_t = t0 - (t0 - 100) = 100s -> v = 1.5 m/s (feasible!)
        "embedding": emb_target,
        "active_attributes": ["backpack", "upper_black", "lower_blue"]
    }

    # Candidate 2: Visual match at distant camera requiring impossible 30 m/s travel
    cand_impossible_speed = {
        "track_id": "TRK-IMPOSSIBLE-SPEED",
        "camera_id": "CAM-001",
        "t_in": t0 - 15.0,
        "t_out": t0 - 5.0,  # delta_t = 5s for 150m = 30 m/s!
        "embedding": emb_target,
        "active_attributes": ["backpack", "upper_black"]
    }

    # Candidate 3: Kinematically valid but completely different person (low Re-ID score)
    emb_diff = [0.0] * 512
    emb_diff[2] = 1.0  # orthogonal
    cand_wrong_person = {
        "track_id": "TRK-WRONG-PERSON",
        "camera_id": "CAM-001",
        "t_in": t0 - 160.0,
        "t_out": t0 - 110.0,
        "embedding": emb_diff,
        "active_attributes": ["handbag", "upper_red"]
    }

    candidate_pool = [cand_genuine, cand_impossible_speed, cand_wrong_person]

    path = tracker.find_backward_origin(incident_probe, candidate_pool)

    # Reconstructed path should start at probe and trace back to CAM-001 genuine tracklet
    assert len(path) == 2
    assert path[0]["track_id"] == "PROBE-CRIME-SCENE"
    assert path[1]["track_id"] == "TRK-ORIGIN-CAM001"
    assert path[1]["camera_id"] == "CAM-001"


def test_retrograde_multi_hop_traversal():
    # 3-hop corridor: CAM-A (origin) -> CAM-B (junction) -> CAM-C (incident)
    distances = {
        "CAM-A": {"CAM-B": 100.0},
        "CAM-B": {"CAM-A": 100.0, "CAM-C": 100.0},
        "CAM-C": {"CAM-B": 100.0}
    }
    tracker = RetrogradeTracker(camera_distances=distances, v_min=0.5, v_max=3.0)

    emb = [0.0] * 512
    emb[0] = 1.0

    t_now = 5000.0
    probe_c = {"track_id": "TRK-C", "camera_id": "CAM-C", "t_in": t_now, "t_out": t_now + 10, "embedding": emb, "active_attributes": ["backpack"]}
    trk_b = {"track_id": "TRK-B", "camera_id": "CAM-B", "t_in": t_now - 100, "t_out": t_now - 70, "embedding": emb, "active_attributes": ["backpack"]}
    trk_a = {"track_id": "TRK-A", "camera_id": "CAM-A", "t_in": t_now - 200, "t_out": t_now - 150, "embedding": emb, "active_attributes": ["backpack"]}

    path = tracker.find_backward_origin(probe_c, [trk_b, trk_a])
    assert len(path) == 3
    assert path[0]["camera_id"] == "CAM-C"
    assert path[1]["camera_id"] == "CAM-B"
    assert path[2]["camera_id"] == "CAM-A"  # Successfully traced back to point of origin!
