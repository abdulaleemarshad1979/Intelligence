"""Unit tests for Transparent Multi-Modal Evidence Evaluators."""

import pytest
import numpy as np
from app.evidence.face import FaceEvidenceEvaluator
from app.evidence.body import BodyEvidenceEvaluator
from app.evidence.gait import GaitEvidenceEvaluator
from app.evidence.pose import PoseEvidenceEvaluator
from app.evidence.clothing import ClothingEvidenceEvaluator
from app.evidence.height import HeightEvidenceEvaluator
from app.evidence.object_features import CarriedObjectEvidenceEvaluator
from app.association.evidence_fusion import EvidenceFusionEngine

def test_face_evidence_unavailable_resilience():
    evaluator = FaceEvidenceEvaluator()
    probe = {"face_status": "UNAVAILABLE", "face_embedding": []}
    cand = {"face_status": "UNAVAILABLE", "face_embedding": []}

    res = evaluator.evaluate(probe, cand)
    assert res["available"] is False
    assert res["grade"] == "UNAVAILABLE"
    assert "turned away" in res["summary"].lower()

def test_body_evidence_strong_match():
    evaluator = BodyEvidenceEvaluator()
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.95, 0.1, 0.0]
    probe = {"body_embedding": v1, "torso_leg_ratio": 0.85}
    cand = {"body_embedding": v2, "torso_leg_ratio": 0.84}

    res = evaluator.evaluate(probe, cand)
    assert res["available"] is True
    assert res["grade"] in ["STRONG", "MODERATE"]
    assert res["score"] > 0.80

def test_gait_evidence_cadence_and_waveform():
    evaluator = GaitEvidenceEvaluator()
    probe = {"cadence_steps_per_sec": 1.8, "spine_tilt_deg": 4.0, "gait_embedding": [1.0, 0.5, 0.2]}
    cand = {"cadence_steps_per_sec": 1.78, "spine_tilt_deg": 4.2, "gait_embedding": [1.0, 0.5, 0.2]}

    res = evaluator.evaluate(probe, cand)
    assert res["available"] is True
    assert res["grade"] in ["STRONG", "MODERATE"]

def test_clothing_evidence_color_palette():
    evaluator = ClothingEvidenceEvaluator()
    probe = {"clothing_upper": "#1b2430", "clothing_lower": "#2c3539"}
    cand = {"clothing_upper": "#1b2430", "clothing_lower": "#2c3539"}

    res = evaluator.evaluate(probe, cand)
    assert res["available"] is True
    assert res["grade"] == "STRONG"
    assert res["score"] >= 0.90

def test_height_evidence_tolerance():
    evaluator = HeightEvidenceEvaluator()
    probe = {"height_cm": 178.0}
    cand = {"height_cm": 177.5}

    res = evaluator.evaluate(probe, cand)
    assert res["available"] is True
    assert res["grade"] == "STRONG"
    assert "consistent" in res["summary"].lower()

def test_carried_object_backpack_detected():
    evaluator = CarriedObjectEvidenceEvaluator()
    probe = {"carried_objects": ["backpack"]}
    cand = {"carried_objects": ["backpack"]}

    res = evaluator.evaluate(probe, cand)
    assert res["available"] is True
    assert res["grade"] == "STRONG"
    assert "backpack" in res["summary"].lower()

def test_dynamic_evidence_fusion_resilient_to_missing_face():
    fusion = EvidenceFusionEngine()
    breakdown = {
        "face": {"available": False, "score": 0.0},
        "body": {"available": True, "score": 0.88},
        "gait": {"available": True, "score": 0.82},
        "clothing": {"available": True, "score": 0.90},
        "height": {"available": True, "score": 0.95},
        "carried_objects": {"available": True, "score": 1.0}
    }
    result = fusion.fuse(breakdown)
    assert result["face_available"] is False
    assert result["composite_score"] >= 0.80
    assert "face" not in result["active_weights"]
    assert sum(result["active_weights"].values()) == pytest.approx(1.0, 0.02)
