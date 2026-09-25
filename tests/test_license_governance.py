"""Tests for Production Framework Governance & Model Recommendations Table."""

import pytest
from app.adapters.license_audit import LicenseAuditor
from fastapi.testclient import TestClient
from app.main import app


def test_framework_governance_table_content():
    auditor = LicenseAuditor()
    table = auditor.get_framework_governance_table()

    assert len(table) == 4

    rows_by_comp = {r["component"]: r for r in table}

    # 1. Object Detection: RT-DETR | Apache 2.0 | Approved
    rtdetr = rows_by_comp["Object Detection"]
    assert rtdetr["recommended_framework"] == "RT-DETR"
    assert rtdetr["license"] == "Apache 2.0"
    assert rtdetr["production_status"] == "Approved"
    assert "AGPL-3.0 copyleft exposure" in rtdetr["rationale"]

    # 2. Pose Estimation: RTMPose (MMPose) | Apache 2.0 | Approved
    pose = rows_by_comp["Pose Estimation"]
    assert pose["recommended_framework"] == "RTMPose (MMPose)"
    assert pose["license"] == "Apache 2.0"
    assert pose["production_status"] == "Approved"
    assert "Sub-millisecond latency" in pose["rationale"]

    # 3. Gait Signature: Handcrafted Kinematics | Proprietary IP | Approved
    gait = rows_by_comp["Gait Signature"]
    assert gait["recommended_framework"] == "Handcrafted Kinematics"
    assert gait["license"] == "Proprietary IP"
    assert gait["production_status"] == "Approved"
    assert "joint angle velocities" in gait["rationale"]
    assert "Zero 3rd-party licensing risk" in gait["rationale"]

    # 4. Deep Gait Models: OpenGait (GaitSet / DeepGait) | Research / Proprietary | Quarantine
    deep_gait = rows_by_comp["Deep Gait Models"]
    assert deep_gait["recommended_framework"] == "OpenGait (GaitSet / DeepGait)"
    assert deep_gait["license"] == "Research / Proprietary"
    assert deep_gait["production_status"] == "Quarantine"
    assert "Quarantine" in deep_gait["rationale"]


def test_governance_api_endpoints():
    client = TestClient(app)

    # 1. Frameworks API
    res = client.get("/api/governance/frameworks")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 4
    components = [item["component"] for item in data]
    assert "Object Detection" in components
    assert "Pose Estimation" in components
    assert "Gait Signature" in components
    assert "Deep Gait Models" in components

    # 2. Section 3 Disparity Veto API
    res_veto = client.get("/api/fusion/disparity-veto")
    assert res_veto.status_code == 200
    veto_data = res_veto.json()
    assert veto_data["architecture_section"] == "3. Signal Fusion Architecture & Disparity Veto"
    assert "soft biometrics alone yield unacceptably high false-match rates" in veto_data["core_principle"]
    assert veto_data["pruning_gates"]["max_height_disparity_cm"] == 12.0
