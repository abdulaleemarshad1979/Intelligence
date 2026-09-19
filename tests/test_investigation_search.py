"""Unit tests for Gotham Person Search & Candidate Association."""

import pytest
import yaml
import os
from app.database.database import init_db
from app.database.repository import Repository
from app.investigation.demo_seed import seed_gotham_demo
from app.search.person_search import PersonSearchCoordinator

def test_gotham_person_search_workflow(tmp_path):
    db_file = str(tmp_path / "test_gotham_search.db")
    init_db(db_file)
    repo = Repository(db_file)

    # Seed the reference scenario
    seed_gotham_demo(repo)

    # Load camera configs
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "cameras.yaml")
    with open(cfg_path, "r") as f:
        camera_registry = yaml.safe_load(f).get("cameras", {})

    coordinator = PersonSearchCoordinator(repo, camera_registry)

    # Probe is Track 481 at CAM-017 18:42:11
    associations = coordinator.find_associated_tracks(probe_track_id="481")

    # Should discover Tracks 774, 991, 144
    found_ids = [a.candidate_track_id for a in associations]
    assert "774" in found_ids
    assert "991" in found_ids
    assert "144" in found_ids

    # Check candidate association evidence breakdown on Track 774
    assoc_774 = next(a for a in associations if a.candidate_track_id == "774")
    assert assoc_774.candidate_camera == "CAM-018"
    assert assoc_774.is_physically_feasible is True

    # Check that individual evidence modalities are transparent
    ev = assoc_774.evidence_breakdown
    assert ev["face"]["available"] is False
    assert ev["body"]["available"] is True
    assert ev["body"]["grade"] in ["STRONG", "MODERATE"]
    assert ev["clothing"]["grade"] in ["STRONG", "SUPPORTING"]
    assert ev["travel_time"]["is_feasible"] is True
    assert ev["trajectory"]["consistent"] is True

    # Check narrative formatting
    narrative = assoc_774.qualitative_narrative
    assert "Face: unavailable" in narrative
    assert "Body:" in narrative
    assert "Trajectory: consistent" in narrative
