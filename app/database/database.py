"""Database connection and initialization module."""

import sqlite3
import os
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "police_records.db")

def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    target_path = db_path or DB_PATH
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: Optional[str] = None):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # 1. Criminal Records Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS criminal_records (
        id TEXT PRIMARY KEY,
        fir_no TEXT NOT NULL,
        unit_name TEXT,
        subdivision TEXT,
        police_station TEXT,
        accused_name TEXT NOT NULL,
        alias TEXT,
        age INTEGER,
        gender TEXT,
        acts_sec TEXT,
        brief_facts TEXT,
        latitude REAL,
        longitude REAL,
        status_of_case TEXT,
        photo_url TEXT,
        known_height_cm REAL,
        torso_leg_ratio REAL,
        stride_length_cm REAL,
        posture_lean_angle REAL,
        posture_correctness REAL,
        clothing_upper_color TEXT,
        clothing_lower_color TEXT,
        face_embedding TEXT,
        body_embedding TEXT,
        gait_embedding TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # 2. Tracks Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tracks (
        track_id TEXT PRIMARY KEY,
        camera_id TEXT NOT NULL,
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
        gait_embedding TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 3. Match Events Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS match_events (
        event_id TEXT PRIMARY KEY,
        track_id TEXT NOT NULL,
        camera_id TEXT NOT NULL,
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

    # 4. Audit Log
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL,
        action TEXT,
        details TEXT,
        operator TEXT DEFAULT 'OFFICER_IN_CHARGE'
    )
    """)

    # 5. Gotham Ontology: Persons / Targets
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS persons (
        person_id TEXT PRIMARY KEY,
        target_code TEXT NOT NULL,
        canonical_name TEXT DEFAULT 'UNIDENTIFIED_TARGET',
        status TEXT DEFAULT 'PERSON_OF_INTEREST',
        notes TEXT DEFAULT '',
        created_at REAL,
        updated_at REAL
    )
    """)

    # 6. Gotham Ontology: Observations
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS observations (
        observation_id TEXT PRIMARY KEY,
        track_id TEXT NOT NULL,
        camera_id TEXT NOT NULL,
        timestamp REAL NOT NULL,
        frame_number INTEGER DEFAULT 0,
        crop_path TEXT DEFAULT '',
        bbox_json TEXT DEFAULT '{}',
        created_at REAL
    )
    """)

    # 7. Gotham Ontology: Features
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS features (
        feature_id TEXT PRIMARY KEY,
        track_id TEXT NOT NULL,
        observation_id TEXT,
        face_status TEXT DEFAULT 'UNAVAILABLE',
        face_embedding TEXT DEFAULT '[]',
        body_embedding TEXT DEFAULT '[]',
        gait_embedding TEXT DEFAULT '[]',
        pose_json TEXT DEFAULT '{}',
        clothing_json TEXT DEFAULT '{}',
        height_cm REAL DEFAULT 0.0,
        carried_objects_json TEXT DEFAULT '[]',
        direction TEXT DEFAULT 'UNKNOWN',
        created_at REAL
    )
    """)

    # 8. Gotham Ontology: Incidents
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS incidents (
        incident_id TEXT PRIMARY KEY,
        case_number TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT DEFAULT '',
        camera_id TEXT NOT NULL,
        incident_time REAL NOT NULL,
        status TEXT DEFAULT 'OPEN',
        priority TEXT DEFAULT 'HIGH',
        officer_in_charge TEXT DEFAULT 'AP-EG-8821',
        seed_track_id TEXT,
        created_at REAL
    )
    """)

    # 9. Gotham Ontology: Cameras
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cameras (
        camera_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        zone TEXT DEFAULT 'Central Division',
        view_direction TEXT DEFAULT 'NORTH',
        connected_topology_json TEXT DEFAULT '[]',
        is_active INTEGER DEFAULT 1,
        ip_address TEXT DEFAULT '127.0.0.1',
        rtsp_url TEXT DEFAULT '',
        manufacturer TEXT DEFAULT 'Generic ONVIF',
        model_name TEXT DEFAULT 'IP Camera',
        mac_address TEXT DEFAULT '',
        discovery_status TEXT DEFAULT 'APPROVED'
    )
    """)

    # Safe column migration for existing databases
    try:
        existing_cols = [r[1] for r in cursor.execute("PRAGMA table_info(cameras)").fetchall()]
        new_cols = [
            ("ip_address", "TEXT DEFAULT '127.0.0.1'"),
            ("rtsp_url", "TEXT DEFAULT ''"),
            ("manufacturer", "TEXT DEFAULT 'Generic ONVIF'"),
            ("model_name", "TEXT DEFAULT 'IP Camera'"),
            ("mac_address", "TEXT DEFAULT ''"),
            ("discovery_status", "TEXT DEFAULT 'APPROVED'")
        ]
        for col_name, col_def in new_cols:
            if col_name not in existing_cols:
                cursor.execute(f"ALTER TABLE cameras ADD COLUMN {col_name} {col_def}")
    except Exception:
        pass

    # 10. Gotham Ontology: Events
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS events (
        event_id TEXT PRIMARY KEY,
        incident_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        description TEXT NOT NULL,
        timestamp REAL NOT NULL,
        metadata_json TEXT DEFAULT '{}'
    )
    """)

    # 11. Gotham Ontology: Relationships (Links)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS relationships (
        relationship_id TEXT PRIMARY KEY,
        source_type TEXT NOT NULL,
        source_id TEXT NOT NULL,
        target_type TEXT NOT NULL,
        target_id TEXT NOT NULL,
        relationship_type TEXT NOT NULL,
        confidence_score REAL DEFAULT 0.0,
        evidence_json TEXT DEFAULT '{}',
        status TEXT DEFAULT 'CANDIDATE',
        created_at REAL
    )
    """)

    # 12. Gotham Ontology: Human Adjudication Reviews
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reviews (
        review_id TEXT PRIMARY KEY,
        relationship_id TEXT,
        target_id TEXT,
        reviewer_badge TEXT NOT NULL,
        reviewer_name TEXT NOT NULL,
        decision TEXT NOT NULL,
        review_notes TEXT DEFAULT '',
        timestamp REAL NOT NULL
    )
    """)

    # Add optional migration columns to tracks if not present
    cursor.execute("PRAGMA table_info(tracks)")
    columns = [col[1] for col in cursor.fetchall()]
    if "start_time" not in columns:
        cursor.execute("ALTER TABLE tracks ADD COLUMN start_time REAL DEFAULT 0.0")
    if "end_time" not in columns:
        cursor.execute("ALTER TABLE tracks ADD COLUMN end_time REAL DEFAULT 0.0")
    if "direction" not in columns:
        cursor.execute("ALTER TABLE tracks ADD COLUMN direction TEXT DEFAULT 'UNKNOWN'")
    if "status" not in columns:
        cursor.execute("ALTER TABLE tracks ADD COLUMN status TEXT DEFAULT 'ACTIVE'")

    conn.commit()
    conn.close()
