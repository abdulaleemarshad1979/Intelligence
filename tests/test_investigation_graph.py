"""Unit tests for Investigation Timeline & Node-Edge Graph Generator."""

import pytest
import time
from app.investigation.timeline import TimelineGenerator
from app.investigation.graph import InvestigationGraphBuilder

def test_timeline_generation():
    generator = TimelineGenerator()
    seed_track = {
        "track_id": "481",
        "camera_id": "CAM-017",
        "first_seen": 1774012931.0
    }
    candidate_associations = [
        {
            "candidate_track_id": "774",
            "candidate_camera": "CAM-018",
            "candidate_time": 1774013345.0,
            "time_delta_sec": 414.0,
            "distance_meters": 420.0,
            "transit_speed_mps": 1.01,
            "is_physically_feasible": True,
            "qualitative_narrative": "Body strong evidence | Trajectory consistent"
        },
        {
            "candidate_track_id": "991",
            "candidate_camera": "CAM-021",
            "candidate_time": 1774013790.0,
            "time_delta_sec": 445.0,
            "distance_meters": 450.0,
            "transit_speed_mps": 1.01,
            "is_physically_feasible": True,
            "qualitative_narrative": "Gait consistent | Direction North"
        }
    ]

    timeline = generator.generate_timeline(candidate_associations, seed_track=seed_track)
    assert len(timeline) == 3
    assert timeline[0]["camera_id"] == "CAM-017"
    assert timeline[1]["camera_id"] == "CAM-018"
    assert timeline[2]["camera_id"] == "CAM-021"
    assert timeline[1]["is_feasible"] is True

def test_investigation_graph_builder():
    builder = InvestigationGraphBuilder()
    incident = {
        "incident_id": "INC-2026-0041",
        "case_number": "INC-2026-0041",
        "camera_id": "CAM-017",
        "seed_track_id": "481"
    }
    candidate_associations = [
        {
            "probe_track_id": "481",
            "candidate_track_id": "774",
            "probe_camera": "CAM-017",
            "candidate_camera": "CAM-018",
            "candidate_time": 1774013345.0,
            "composite_score": 0.88,
            "association_status": "CANDIDATE"
        }
    ]

    graph = builder.build_graph(incident, candidate_associations, relationships=[])
    assert len(graph["nodes"]) >= 4  # Incident, CAM-017, Track 481, CAM-018, Track 774
    assert len(graph["edges"]) >= 3
    node_types = {n["type"] for n in graph["nodes"]}
    assert "INCIDENT" in node_types
    assert "CAMERA" in node_types
    assert "TRACK" in node_types
