"""SQLite schema for the casework layer (cases, media custody, reviews, camera inventory)."""

import os
import sqlite3
from typing import Optional

from app.database.database import get_db_connection

CASEWORK_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "casework.db"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS cw_cameras (
    camera_id TEXT PRIMARY KEY,
    sheet_serial INTEGER,
    ip_address TEXT NOT NULL UNIQUE,
    camera_type TEXT NOT NULL,
    location TEXT NOT NULL,
    police_station TEXT,
    sub_division TEXT,
    latitude REAL,
    longitude REAL,
    from_side TEXT,
    towards TEXT,
    data_issues TEXT NOT NULL DEFAULT '[]',
    source_file TEXT,
    imported_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cw_cameras_ps ON cw_cameras (police_station);
CREATE INDEX IF NOT EXISTS idx_cw_cameras_type ON cw_cameras (camera_type);

CREATE TABLE IF NOT EXISTS cw_cases (
    case_id TEXT PRIMARY KEY,
    case_number TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    scenario TEXT NOT NULL CHECK (scenario IN ('PHOTO', 'VIDEO')),
    status TEXT NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'IN_REVIEW', 'CLOSED')),
    notes TEXT NOT NULL DEFAULT '',
    created_by TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS cw_media (
    media_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cw_cases (case_id),
    kind TEXT NOT NULL CHECK (kind IN ('IMAGE', 'VIDEO')),
    role TEXT NOT NULL CHECK (role IN ('ORIGINAL', 'DERIVED', 'PROXY')),
    parent_media_id TEXT REFERENCES cw_media (media_id),
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    storage_path TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    mime_type TEXT,
    width INTEGER,
    height INTEGER,
    duration_s REAL,
    nominal_fps REAL,
    is_vfr INTEGER,
    label TEXT NOT NULL DEFAULT '',
    source_description TEXT NOT NULL DEFAULT '',
    last_known_time REAL,
    last_known_location TEXT,
    last_known_latitude REAL,
    last_known_longitude REAL,
    source_camera_id TEXT REFERENCES cw_cameras (camera_id),
    added_by TEXT NOT NULL,
    added_at REAL NOT NULL,
    CHECK ((role = 'ORIGINAL') = (parent_media_id IS NULL))
);
CREATE INDEX IF NOT EXISTS idx_cw_media_case ON cw_media (case_id, role);
CREATE INDEX IF NOT EXISTS idx_cw_media_sha ON cw_media (sha256);
CREATE INDEX IF NOT EXISTS idx_cw_media_parent ON cw_media (parent_media_id);

CREATE TABLE IF NOT EXISTS cw_derivations (
    derivation_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cw_cases (case_id),
    parent_media_id TEXT NOT NULL REFERENCES cw_media (media_id),
    result_media_id TEXT NOT NULL REFERENCES cw_media (media_id),
    step_index INTEGER NOT NULL,
    operation TEXT NOT NULL,
    parameters TEXT NOT NULL DEFAULT '{}',
    tool TEXT NOT NULL DEFAULT '',
    performed_by TEXT NOT NULL,
    performed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cw_deriv_result ON cw_derivations (result_media_id);

CREATE TABLE IF NOT EXISTS cw_frame_times (
    media_id TEXT PRIMARY KEY REFERENCES cw_media (media_id),
    pts_seconds TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cw_quality_notes (
    note_id TEXT PRIMARY KEY,
    media_id TEXT NOT NULL REFERENCES cw_media (media_id),
    origin TEXT NOT NULL CHECK (origin IN ('AUTO', 'MANUAL')),
    metrics TEXT NOT NULL DEFAULT '{}',
    warnings TEXT NOT NULL DEFAULT '[]',
    note TEXT NOT NULL DEFAULT '',
    author TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS cw_annotations (
    annotation_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cw_cases (case_id),
    media_id TEXT NOT NULL REFERENCES cw_media (media_id),
    kind TEXT NOT NULL CHECK (kind IN ('INTERVAL', 'PERSON', 'REGION', 'NOTE')),
    start_s REAL,
    end_s REAL,
    bbox TEXT,
    label TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    author TEXT NOT NULL,
    created_at REAL NOT NULL,
    CHECK (start_s IS NULL OR end_s IS NULL OR end_s >= start_s)
);
CREATE INDEX IF NOT EXISTS idx_cw_annot_media ON cw_annotations (media_id, kind);

CREATE TABLE IF NOT EXISTS cw_clip_candidates (
    candidate_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cw_cases (case_id),
    camera_id TEXT NOT NULL REFERENCES cw_cameras (camera_id),
    media_id TEXT REFERENCES cw_media (media_id),
    window_start REAL NOT NULL,
    window_end REAL NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ('SPATIOTEMPORAL', 'MANUAL')),
    rationale TEXT NOT NULL DEFAULT '',
    created_by TEXT NOT NULL,
    created_at REAL NOT NULL,
    CHECK (window_end >= window_start),
    UNIQUE (case_id, camera_id, window_start, window_end)
);

CREATE TABLE IF NOT EXISTS cw_review_decisions (
    decision_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cw_cases (case_id),
    candidate_id TEXT NOT NULL REFERENCES cw_clip_candidates (candidate_id),
    decision TEXT NOT NULL CHECK (decision IN ('ACCEPT', 'REJECT', 'DEFER')),
    reviewer_badge TEXT NOT NULL,
    reviewer_name TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    decided_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cw_review_cand ON cw_review_decisions (candidate_id, decided_at);

-- Custody: originals and the history of what was done to them cannot be edited or removed.
CREATE TRIGGER IF NOT EXISTS trg_cw_media_immutable
BEFORE UPDATE OF case_id, kind, role, parent_media_id, sha256, size_bytes, storage_path,
    original_filename, source_description, last_known_time, last_known_location,
    last_known_latitude, last_known_longitude, source_camera_id, added_by, added_at
ON cw_media
BEGIN SELECT RAISE(ABORT, 'cw_media custody fields are immutable'); END;

CREATE TRIGGER IF NOT EXISTS trg_cw_media_no_delete
BEFORE DELETE ON cw_media
BEGIN SELECT RAISE(ABORT, 'cw_media rows cannot be deleted'); END;

CREATE TRIGGER IF NOT EXISTS trg_cw_deriv_no_update
BEFORE UPDATE ON cw_derivations
BEGIN SELECT RAISE(ABORT, 'cw_derivations is append-only'); END;

CREATE TRIGGER IF NOT EXISTS trg_cw_deriv_no_delete
BEFORE DELETE ON cw_derivations
BEGIN SELECT RAISE(ABORT, 'cw_derivations is append-only'); END;

CREATE TRIGGER IF NOT EXISTS trg_cw_review_no_update
BEFORE UPDATE ON cw_review_decisions
BEGIN SELECT RAISE(ABORT, 'cw_review_decisions is append-only'); END;

CREATE TRIGGER IF NOT EXISTS trg_cw_review_no_delete
BEFORE DELETE ON cw_review_decisions
BEGIN SELECT RAISE(ABORT, 'cw_review_decisions is append-only'); END;
"""


def connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    conn = get_db_connection(db_path or CASEWORK_DB_PATH)
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_casework_db(db_path: Optional[str] = None) -> str:
    path = db_path or CASEWORK_DB_PATH
    conn = connect(path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
    return path
