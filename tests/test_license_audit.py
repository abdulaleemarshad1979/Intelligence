"""Unit tests for Open-Source License and Governance Audit."""

import pytest
from app.adapters.license_audit import LicenseAuditor, MODEL_LICENSE_CATALOG


def test_license_catalog_contents():
    auditor = LicenseAuditor(MODEL_LICENSE_CATALOG)
    catalog = auditor.get_full_catalog()
    assert len(catalog) >= 10

    # Verify OpenGait has academic restriction
    opengait_info = auditor.get_info("opengait")
    assert not opengait_info.is_commercial_ready
    assert "Academic" in opengait_info.code_license

    # Verify InsightFace code is MIT but weights are restricted
    insightface_info = auditor.get_info("insightface")
    assert insightface_info.code_license == "MIT"
    assert not insightface_info.is_commercial_ready

    # Verify DeepStream, FastReID, OSNet are commercial ready
    assert auditor.get_info("deepstream").is_commercial_ready
    assert auditor.get_info("fastreid").is_commercial_ready
    assert auditor.get_info("osnet").is_commercial_ready


def test_audit_active_stack():
    auditor = LicenseAuditor(MODEL_LICENSE_CATALOG)

    # 1. Commercial-ready stack
    commercial_stack = {
        "detection": "rtdetr",
        "tracking": "botsort",
        "reid": "osnet",
        "pose": "rtmpose",
        "gait": "gaitset"
    }
    res_comm = auditor.audit_active_stack(commercial_stack)
    assert res_comm["overall_status"] == "COMMERCIAL_PRODUCTION_READY"
    assert res_comm["commercial_ready_count"] == 5
    assert len(res_comm["academic_restricted_components"]) == 0

    # 2. Research-restricted stack (OpenGait or InsightFace)
    research_stack = {
        "detection": "yolo",
        "tracking": "bytetrack",
        "reid": "fastreid",
        "face": "insightface",
        "gait": "opengait"
    }
    res_res = auditor.audit_active_stack(research_stack)
    assert res_res["overall_status"] == "RESEARCH_ONLY"
    assert len(res_res["academic_restricted_components"]) >= 1
