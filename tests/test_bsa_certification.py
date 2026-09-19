"""Unit tests for Section 63 Bharatiya Sakshya Adhiniyam (BSA), 2023 Evidentiary Ledger.

Tests:
- SHA-256 cryptographic hashing of raw evidentiary video vs restored candidate crops
- Section 63(4) Part A (Custodian) & Part B (Cyber Forensic Examiner) certificate generation
- Tamper detection on modified electronic records
- NTP Stratum-1 clock synchronization verification
- Cryptographic hash chaining in append-only audit ledger
"""

import pytest
import hashlib
from app.integrations.bsa_evidence import BSAEvidenceLedger


@pytest.fixture
def bsa_engine():
    return BSAEvidenceLedger()


def test_sha256_generation(bsa_engine):
    """Verify SHA-256 cryptographic digest calculation."""
    sample_bytes = b"MUNICIPAL_CCTV_STREAM_SURVEILLANCE_FRAME_RAW"
    expected = hashlib.sha256(sample_bytes).hexdigest()
    assert bsa_engine.generate_sha256(sample_bytes) == expected


def test_section_63_bsa_certificate_structure(bsa_engine):
    """Verify statutory certificate generation under Section 63 BSA, 2023."""
    raw_frame = b"RAW_PIXEL_MATRIX_FRAME_1092"
    enhanced_crop = b"CODEFORMER_RESTORED_FACE_CROP_1092"

    fir_dossier = {
        "fir_no": "FIR-2026-AP-0194",
        "ps_code": "PS-KAKINADA-PORT",
        "bns_sections": "BNS Section 303(2), Section 111",
        "threat_level": "CATEGORY_A_CRITICAL"
    }

    operators = ["DESK_OFFICER_AP_4412", "SUPV_AP_1002"]

    cert = bsa_engine.build_certificate(
        incident_id="INC-2026-0041",
        camera_id="CAM-002",
        raw_frame_bytes=raw_frame,
        enhanced_frame_bytes=enhanced_crop,
        matched_fir_dossier=fir_dossier,
        operator_ids=operators
    )

    # Validate statutory legal framework reference
    assert cert["legal_framework"] == "Section 63 Bharatiya Sakshya Adhiniyam, 2023"
    assert "63(4)" in cert["section_subclause"]

    # Validate dual hash preservation
    crypto = cert["cryptographic_integrity"]
    assert crypto["hashing_algorithm"] == "SHA-256"
    assert crypto["raw_evidentiary_hash"] == hashlib.sha256(raw_frame).hexdigest()
    assert crypto["enhanced_crop_hash"] == hashlib.sha256(enhanced_crop).hexdigest()
    assert crypto["raw_evidentiary_hash"] != crypto["enhanced_crop_hash"]  # Dual hashes must be distinct

    # Validate NTP clock synchronization
    cam_meta = cert["camera_metadata"]
    assert cam_meta["ntp_synchronized"] is True
    assert cam_meta["clock_drift_ms"] < 5.0

    # Validate dual-operator sign-off attestation
    attestation = cert["sign_off_attestation"]
    assert attestation["part_a_custodian_id"] == "DESK_OFFICER_AP_4412"
    assert attestation["part_b_forensic_examiner_id"] == "SUPV_AP_1002"
    assert attestation["dual_operator_confirmed"] is True


def test_tamper_detection_verification(bsa_engine):
    """Test cryptographic verification detecting tampered frame bytes."""
    raw_frame = b"ORIGINAL_AUTHENTIC_CCTV_RECORD"
    enhanced_crop = b"CODEFORMER_RESTORED_RECORD"

    cert = bsa_engine.build_certificate(
        incident_id="INC-TEST-TAMPER",
        camera_id="CAM-001",
        raw_frame_bytes=raw_frame,
        enhanced_frame_bytes=enhanced_crop,
        matched_fir_dossier={"fir_no": "FIR-123"},
        operator_ids=["OP-1", "OP-2"]
    )

    # 1. Verification should pass on authentic bytes
    assert bsa_engine.verify_certificate(cert, raw_frame, enhanced_crop) is True

    # 2. Tampered raw frame byte flip should fail
    tampered_raw = b"TAMPERED_AUTHENTIC_CCTV_RECORD"
    assert bsa_engine.verify_certificate(cert, tampered_raw, enhanced_crop) is False

    # 3. Tampered enhanced frame should fail
    tampered_enh = b"TAMPERED_RESTORED_RECORD"
    assert bsa_engine.verify_certificate(cert, raw_frame, tampered_enh) is False


def test_immutable_ledger_hash_chaining(bsa_engine):
    """Test cryptographic hash chaining across consecutive certificates in audit ledger."""
    cert1 = bsa_engine.build_certificate(
        incident_id="INC-001", camera_id="CAM-1",
        raw_frame_bytes=b"RAW1", enhanced_frame_bytes=b"ENH1",
        matched_fir_dossier={}, operator_ids=["OP1"]
    )
    cert2 = bsa_engine.build_certificate(
        incident_id="INC-002", camera_id="CAM-2",
        raw_frame_bytes=b"RAW2", enhanced_frame_bytes=b"ENH2",
        matched_fir_dossier={}, operator_ids=["OP2"]
    )

    # cert2's previous block hash must match cert1's certificate digest
    assert cert2["ledger_chain"]["previous_block_hash"] == cert1["certificate_digest"]
    assert cert2["ledger_chain"]["sequence_number"] == 2
