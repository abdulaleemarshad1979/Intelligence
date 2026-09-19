"""Unit tests for Gotham Investigation Ontology Models & Database Tables."""

import pytest
import time
from app.database.models import (
    PersonTarget, CameraEntity, IncidentCase,
    ObservationRecord, FeatureRecord, RelationshipLink, AdjudicationReview
)
from app.ontology.person import TargetProfile
from app.ontology.track import TrackEntity
from app.ontology.incident import IncidentEntity
from app.database.repository import Repository

def test_ontology_models_instantiation():
    person = PersonTarget(
        person_id="POI-TEST-001",
        target_code="TARGET-001",
        canonical_name="TEST_SUBJECT"
    )
    assert person.person_id == "POI-TEST-001"
    assert person.status == "PERSON_OF_INTEREST"

    cam = CameraEntity(
        camera_id="CAM-017",
        name="Sector 4 Exit",
        latitude=16.992,
        longitude=82.245
    )
    assert cam.camera_id == "CAM-017"

    inc = IncidentCase(
        incident_id="INC-TEST-01",
        case_number="INC-2026-0041",
        title="Break-in investigation",
        camera_id="CAM-017",
        seed_track_id="481"
    )
    assert inc.case_number == "INC-2026-0041"
    assert inc.seed_track_id == "481"

def test_ontology_repository_persistence(tmp_path):
    db_file = str(tmp_path / "test_police.db")
    from app.database.database import init_db
    init_db(db_file)
    repo = Repository(db_file)

    # 1. Person
    p = PersonTarget("POI-01", "TARGET-01", "UNKNOWN", "PERSON_OF_INTEREST")
    assert repo.save_person(p) is True
    p_ret = repo.get_person_by_id("POI-01")
    assert p_ret is not None
    assert p_ret.target_code == "TARGET-01"

    # 2. Incident
    inc = IncidentCase("INC-01", "INC-2026-0041", "Test Incident", "Desc", "CAM-017", time.time())
    assert repo.save_incident(inc) is True
    inc_ret = repo.get_incident_by_id("INC-01")
    assert inc_ret is not None
    assert inc_ret.case_number == "INC-2026-0041"

    # 3. Observation
    obs = ObservationRecord("OBS-01", "481", "CAM-017", time.time())
    assert repo.save_observation(obs) is True
    obs_list = repo.get_observations_for_track("481")
    assert len(obs_list) == 1
    assert obs_list[0].camera_id == "CAM-017"

    # 4. Feature
    feat = FeatureRecord("FEAT-01", "481", "OBS-01", face_status="UNAVAILABLE", height_cm=178.0, direction="NORTH")
    assert repo.save_feature(feat) is True
    feat_ret = repo.get_feature_for_track("481")
    assert feat_ret is not None
    assert feat_ret.face_status == "UNAVAILABLE"
    assert feat_ret.height_cm == 178.0

    # 5. Relationship
    rel = RelationshipLink("REL-01", "TRACK", "481", "TRACK", "774", "CANDIDATE_SAME_PERSON", 0.88)
    assert repo.save_relationship(rel) is True
    rels = repo.get_relationships(source_id="481")
    assert len(rels) == 1
    assert rels[0].confidence_score == 0.88
