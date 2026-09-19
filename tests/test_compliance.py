"""Unit tests for Police Governance, Cryptographic Audit Ledger, RBAC, and Human Review Gate."""

import os
import sqlite3
import pytest
import time
from app.compliance.rbac import UserRole, Permission, OfficerIdentity
from app.compliance.audit import AuditLogger
from app.compliance.human_review import HumanReviewGate, ReviewSubmission
from app.compliance.retention import BiometricRetentionEngine

@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test_police.db")
    # Initialize schema
    conn = sqlite3.connect(db_file)
    conn.execute("""
        CREATE TABLE match_events (
            event_id TEXT PRIMARY KEY,
            track_id TEXT,
            camera_id TEXT,
            suspect_id TEXT,
            suspect_name TEXT,
            fir_no TEXT,
            total_confidence REAL,
            face_score REAL,
            body_score REAL,
            gait_score REAL,
            height_score REAL,
            is_face_available INTEGER,
            status TEXT,
            evidence_breakdown TEXT,
            created_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE tracks (
            track_id TEXT PRIMARY KEY,
            camera_id TEXT,
            first_seen REAL,
            last_seen REAL,
            frame_count INTEGER,
            best_frame_path TEXT,
            face_visible INTEGER,
            face_status TEXT,
            face_tier_details TEXT,
            estimated_height_cm REAL,
            body_proportions TEXT,
            clothing_upper TEXT,
            clothing_lower TEXT,
            stride_length_px REAL,
            stride_length_cm REAL,
            cadence_steps_per_sec REAL,
            spine_tilt_deg REAL,
            posture_score REAL,
            gait_wave TEXT,
            face_embedding TEXT,
            body_embedding TEXT,
            gait_embedding TEXT
        )
    """)
    # Insert a sample match event
    conn.execute("""
        INSERT INTO match_events (
            event_id, track_id, camera_id, suspect_id, suspect_name,
            fir_no, total_confidence, face_score, body_score, gait_score,
            height_score, is_face_available, status, evidence_breakdown, created_at
        ) VALUES (
            'EVT-TEST-001', 'TRK-001', 'CAM-001', 'SUS-101', 'K. Ramesh',
            'FIR-204/2026', 0.82, 0.80, 0.85, 0.82, 0.80, 1, 'REVIEW_REQUIRED', '{}', ?
        )
    """, (time.time(),))
    conn.commit()
    conn.close()
    return db_file

def test_rbac_permissions():
    patrol = OfficerIdentity(
        user_id="OFF-01",
        badge_number="BP-001",
        full_name="Patrol Officer",
        role=UserRole.PATROL_OFFICER
    )
    assert patrol.can(Permission.VIEW_LIVE_FEEDS) is True
    assert patrol.can(Permission.SIGN_OFF_MATCH) is False
    assert patrol.can(Permission.VIEW_AUDIT_LOGS) is False

    investigator = OfficerIdentity(
        user_id="OFF-02",
        badge_number="BP-002",
        full_name="Lead Investigator",
        role=UserRole.INVESTIGATOR
    )
    assert investigator.can(Permission.SIGN_OFF_MATCH) is True
    assert investigator.can(Permission.INSPECT_DOSSIER) is True

def test_audit_hash_chain_integrity(temp_db):
    audit = AuditLogger(temp_db)
    
    # Log two consecutive actions
    log1 = audit.log_action("LOGIN", "SYSTEM", {"ip": "10.0.0.1"})
    log2 = audit.log_action("BIOMETRIC_SEARCH", "SUSPECT-101", {"query": "face"})

    assert log1["entry_hash"] == log2["prev_hash"]

    # Verify blockchain integrity
    status = audit.verify_integrity()
    assert status["verified"] is True
    assert status["total_records"] == 2

def test_tamper_detection(temp_db):
    audit = AuditLogger(temp_db)
    audit.log_action("ACTION_1", "RES_1", {"msg": "legit 1"})
    audit.log_action("ACTION_2", "RES_2", {"msg": "legit 2"})

    # Maliciously edit entry in raw SQLite
    conn = sqlite3.connect(temp_db)
    conn.execute("UPDATE cryptographic_audit_ledger SET action_type = 'MALICIOUS_ALTER' WHERE id = 1")
    conn.commit()
    conn.close()

    # Integrity verification must flag the tampering!
    status = audit.verify_integrity()
    assert status["verified"] is False
    assert "Content tampered" in status["reason"]

def test_human_review_gate(temp_db):
    gate = HumanReviewGate(temp_db)
    sub = ReviewSubmission(
        event_id="EVT-TEST-001",
        verdict="CONFIRMED_MATCH",
        investigator_notes="Multi-tier brow symmetry and 172cm height delta verify suspect identity.",
        badge_number="AP-1042",
        officer_name="Inspector Sekhar"
    )
    res = gate.submit_review(sub)
    assert res["success"] is True
    assert res["verdict"] == "CONFIRMED_MATCH"

    # Verify updated row
    conn = sqlite3.connect(temp_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM match_events WHERE event_id = 'EVT-TEST-001'").fetchone()
    conn.close()

    assert row["status"] == "CONFIRMED_MATCH"
    assert row["reviewed_by_officer"] == "Inspector Sekhar"
    assert "Multi-tier" in row["review_notes"]

def test_biometric_retention_status(temp_db, tmp_path):
    tracks_dir = str(tmp_path / "tracks")
    os.makedirs(tracks_dir, exist_ok=True)
    engine = BiometricRetentionEngine(db_path=temp_db, tracks_dir=tracks_dir)
    status = engine.get_retention_status()
    assert "policy" in status
    assert "statistics" in status
    assert status["policy"]["unmatched_retention_hours"] == 72.0
