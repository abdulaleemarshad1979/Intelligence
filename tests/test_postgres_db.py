"""Unit & Integration tests for PostgreSQL Watchlist Database."""

import pytest
import time
import base64
import numpy as np
import cv2

from app.database.postgres import PostgresWatchlistDB, postgres_db
from app.vision.face_watch import LiveFaceWatcher


@pytest.fixture(scope="module")
def pg_db():
    """Ensure PostgreSQL is accessible for tests."""
    status = postgres_db.get_status()
    if not status.get("connected"):
        pytest.skip(f"PostgreSQL not accessible: {status.get('error')}")
    # Start clean
    postgres_db.clear_all()
    yield postgres_db
    # Clean up after module
    postgres_db.clear_all()


def test_postgres_connection_and_status(pg_db):
    status = pg_db.get_status()
    assert status["connected"] is True
    assert "PostgreSQL" in status["database"]
    assert "version" in status
    assert isinstance(status["total_records"], int)
    assert status["ping_ms"] >= 0


def test_postgres_suspect_lifecycle(pg_db):
    # 1. Save new suspect
    synthetic_emb = [0.123] * 128
    suspect_data = {
        "target_id": "TGT-PG-UNIT-01",
        "name": "Vikram Rathore",
        "reference_path": "/data/targets/vikram.jpg",
        "reference_url": "/data/targets/vikram.jpg",
        "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
        "embedding": synthetic_emb,
        "threshold": 0.58,
        "quality": {
            "quality_score": 0.89,
            "laplacian_var": 142.5
        },
        "notes": "Armed and high priority",
        "fir_no": "FIR-2026-AP-999"
    }

    ok = pg_db.save_suspect(suspect_data)
    assert ok is True

    # 2. Retrieve all suspects
    suspects = pg_db.get_all_suspects()
    assert len(suspects) >= 1
    found = next((s for s in suspects if s["target_id"] == "TGT-PG-UNIT-01"), None)
    assert found is not None
    assert found["name"] == "Vikram Rathore"
    assert found["fir_no"] == "FIR-2026-AP-999"
    assert found["notes"] == "Armed and high priority"
    assert len(found["embedding"]) == 128
    assert found["threshold"] == 0.58
    assert found["quality_score"] == 0.89
    assert found["laplacian_var"] == 142.5
    assert found["image_base64"] == "data:image/jpeg;base64,/9j/4AAQSkZJRg=="

    # 3. Update suspect (upsert)
    suspect_data["name"] = "Vikram Rathore (Updated)"
    suspect_data["threshold"] = 0.62
    pg_db.save_suspect(suspect_data)

    suspects_updated = pg_db.get_all_suspects()
    found_updated = next((s for s in suspects_updated if s["target_id"] == "TGT-PG-UNIT-01"), None)
    assert found_updated is not None
    assert found_updated["name"] == "Vikram Rathore (Updated)"
    assert found_updated["threshold"] == 0.62

    # 4. Save capture record
    capture_data = {
        "alert_id": "ALT-PG-TEST-001",
        "target_id": "TGT-PG-UNIT-01",
        "target_name": "Vikram Rathore (Updated)",
        "camera_id": "CAM-NORTH-04",
        "confidence": 0.884,
        "similarity_pct": 88.4,
        "full_frame_path": "/data/captures/full_001.jpg",
        "face_crop_path": "/data/captures/face_001.jpg",
        "body_crop_path": None,
        "full_frame_url": "/data/captures/full_001.jpg",
        "face_crop_url": "/data/captures/face_001.jpg",
        "raw_frame_hash": "a" * 64,
        "face_crop_hash": "b" * 64
    }
    cap_ok = pg_db.save_capture(capture_data)
    assert cap_ok is True

    # 5. Delete suspect
    del_ok = pg_db.delete_suspect("TGT-PG-UNIT-01")
    assert del_ok is True

    suspects_after_del = pg_db.get_all_suspects()
    assert not any(s["target_id"] == "TGT-PG-UNIT-01" for s in suspects_after_del)


def test_live_face_watcher_postgres_hydration(pg_db, tmp_path):
    """Verify that LiveFaceWatcher rehydrates its vector index directly from PostgreSQL."""
    # Seed PostgreSQL with a suspect
    synthetic_emb = [0.05] * 128
    suspect_data = {
        "target_id": "TGT-HYDRATE-01",
        "name": "Hydrated Target",
        "reference_path": "/data/targets/hydrate.jpg",
        "reference_url": "/data/targets/hydrate.jpg",
        "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
        "embedding": synthetic_emb,
        "threshold": 0.50,
        "quality": {"quality_score": 0.8, "laplacian_var": 100.0},
        "notes": "Persisted in Postgres",
        "fir_no": "FIR-HYDRATE"
    }
    pg_db.save_suspect(suspect_data)

    # Initialize a new watcher with sync_db=True
    storage_dir = str(tmp_path / "captures")
    watcher = LiveFaceWatcher(
        storage_dir=storage_dir,
        sync_db=True
    )

    # Verify watcher auto-loaded suspect from PostgreSQL
    assert "TGT-HYDRATE-01" in watcher.targets
    assert watcher.targets["TGT-HYDRATE-01"]["name"] == "Hydrated Target"

    # Verify vector index search can find the hydrated target
    query_emb = np.array(synthetic_emb, dtype=np.float32)
    hits = watcher.vector_index.search(query_emb, top_k=1, threshold=0.40)
    assert len(hits) == 1
    assert hits[0]["target_id"] == "TGT-HYDRATE-01"
    assert hits[0]["similarity"] > 0.99

    # Clean up
    pg_db.delete_suspect("TGT-HYDRATE-01")
