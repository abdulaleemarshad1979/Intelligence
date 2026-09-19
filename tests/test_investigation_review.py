"""Unit tests for Human Adjudication Review Gate & Investigation API."""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database.repository import Repository
from app.investigation.review import HumanAdjudicationGate

client = TestClient(app)

def test_human_adjudication_workflow(tmp_path):
    db_file = str(tmp_path / "test_review.db")
    from app.database.database import init_db
    init_db(db_file)
    repo = Repository(db_file)

    gate = HumanAdjudicationGate(repo)
    res = gate.submit_review(
        relationship_id="REL-481-774",
        reviewer_badge="AP-EG-8821",
        reviewer_name="Inspector R. Varma",
        decision="CONFIRM_IDENTITY",
        review_notes="Appearance and northbound transit corroborated with witness statement."
    )

    assert res["decision"] == "CONFIRM_IDENTITY"
    assert res["status"] == "CONFIRMED"

    # Verify review stored in DB
    reviews = repo.get_reviews()
    assert len(reviews) == 1
    assert reviews[0].reviewer_badge == "AP-EG-8821"

def test_api_investigation_find_person():
    response = client.post("/api/investigation/find_person", json={
        "incident_id": "INC-2026-0041",
        "probe_track_id": "481"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["case_number"] == "INC-2026-0041"
    assert "probe" in data
    assert "movement_timeline" in data
    assert "candidate_associations" in data
    assert "investigation_graph" in data

    # Check that candidate tracks include 774, 991, 144
    cand_ids = [c["candidate_track_id"] for c in data["candidate_associations"]]
    assert "774" in cand_ids
    assert "991" in cand_ids
    assert "144" in cand_ids

def test_api_investigation_review_submission():
    response = client.post("/api/investigation/review", json={
        "relationship_id": "REL-481-774",
        "reviewer_badge": "AP-EG-8821",
        "reviewer_name": "Inspector R. Varma",
        "decision": "CONFIRM_IDENTITY",
        "review_notes": "Confirmed candidate link between CAM-017 and CAM-018"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["adjudication"]["status"] == "CONFIRMED"
