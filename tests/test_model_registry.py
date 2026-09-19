"""Unit tests for Pluggable Model Registry and Dynamic Model Switching."""

import pytest
from app.adapters.registry import ModelRegistry


def test_registry_initialization():
    registry = ModelRegistry()
    assert registry.detector is not None
    assert registry.tracker is not None
    assert registry.reid is not None
    assert registry.face is not None
    assert registry.pose is not None
    assert registry.gait is not None

    status = registry.get_status()
    assert "active_models" in status
    assert "backends" in status
    assert "compliance_audit" in status


def test_registry_model_switching():
    registry = ModelRegistry()

    # Switch detector: yolo -> rtdetr
    res_det = registry.select_model("detection", "rtdetr")
    assert res_det["status"] == "SUCCESS"
    assert registry.active_keys["detection"] == "rtdetr"
    assert registry.detector.get_backend_info()["model_id"] == "rtdetr"

    # Switch tracker: bytetrack -> botsort -> deepstream -> mmtracking
    registry.select_model("tracking", "botsort")
    assert registry.tracker.get_backend_info()["model_id"] == "botsort"

    registry.select_model("tracking", "deepstream")
    assert registry.tracker.get_backend_info()["model_id"] == "deepstream"

    registry.select_model("tracking", "mmtracking")
    assert registry.tracker.get_backend_info()["model_id"] == "mmtracking"

    # Switch reid: osnet -> fastreid
    registry.select_model("reid", "fastreid")
    assert registry.reid.get_backend_info()["model_id"] == "fastreid"

    # Switch gait: gaitset -> opengait
    registry.select_model("gait", "opengait")
    assert registry.gait.get_backend_info()["model_id"] == "opengait"


def test_registry_invalid_switching():
    registry = ModelRegistry()

    with pytest.raises(ValueError, match="Unknown modality category"):
        registry.select_model("telepathy", "model_x")

    with pytest.raises(ValueError, match="Unknown model 'non_existent'"):
        registry.select_model("detection", "non_existent")


def test_get_all_registered_models():
    registry = ModelRegistry()
    all_models = registry.get_all_registered_models()

    assert "detection" in all_models
    assert len(all_models["detection"]) >= 2  # yolo, rtdetr
    assert "tracking" in all_models
    assert len(all_models["tracking"]) >= 4   # bytetrack, botsort, deepstream, mmtracking
    assert "reid" in all_models
    assert len(all_models["reid"]) >= 2       # osnet, fastreid
    assert "gait" in all_models
    assert len(all_models["gait"]) >= 2       # opengait, gaitset
