"""Unit tests for cross-camera correlation and spatio-temporal tracking."""

import pytest
import time
from app.reid.cross_camera import CrossCameraTracker, haversine_distance_meters

@pytest.fixture
def mock_cameras():
    return {
        "CAM-001": {
            "name": "North Entrance",
            "location": "Sector 1",
            "latitude": 16.9890,
            "longitude": 82.2475,
            "adjacent_cameras": ["CAM-002"]
        },
        "CAM-002": {
            "name": "East Gate",
            "location": "Sector 2",
            "latitude": 16.9905,
            "longitude": 82.2490,
            "adjacent_cameras": ["CAM-001", "CAM-003"]
        },
        "CAM-003": {
            "name": "Canal Checkpoint",
            "location": "Sector 3",
            "latitude": 16.9950,
            "longitude": 82.2530,
            "adjacent_cameras": ["CAM-002"]
        }
    }

def test_haversine_distance():
    # Test distance between known points
    d = haversine_distance_meters(16.9890, 82.2475, 16.9905, 82.2490)
    assert 200.0 <= d <= 300.0

def test_travel_window(mock_cameras):
    tracker = CrossCameraTracker(mock_cameras)
    dist_m, min_s, max_s = tracker.compute_travel_window("CAM-001", "CAM-002")
    assert dist_m > 0
    assert min_s > 0
    assert max_s > min_s

def test_cross_camera_correlation(mock_cameras):
    tracker = CrossCameraTracker(mock_cameras)

    now = time.time()
    track_a = {
        "track_id": "TRACK-0001",
        "camera_id": "CAM-001",
        "first_seen": now - 300,
        "last_seen": now - 200,
        "body_embedding": [0.1] * 512,
        "clothing_upper": "#334455",
        "clothing_lower": "#112233",
        "estimated_height_cm": 172.0
    }

    # Track B on adjacent camera after 60 seconds (physically feasible)
    track_b = {
        "track_id": "TRACK-0002",
        "camera_id": "CAM-002",
        "first_seen": now - 140,
        "last_seen": now - 80,
        "body_embedding": [0.1] * 512,
        "clothing_upper": "#334455",
        "clothing_lower": "#112233",
        "estimated_height_cm": 171.5
    }

    link = tracker.correlate_tracks(track_a, track_b)
    assert link.is_physically_feasible is True
    assert link.reid_similarity >= 0.85
    assert link.overall_link_confidence >= 0.80

def test_impossible_teleportation_rejected(mock_cameras):
    tracker = CrossCameraTracker(mock_cameras)
    now = time.time()
    track_a = {
        "track_id": "TRACK-0001",
        "camera_id": "CAM-001",
        "last_seen": now - 10,
        "body_embedding": [0.1] * 512,
        "clothing_upper": "#334455",
        "clothing_lower": "#112233"
    }
    # Track B appears 1 second later across 250m! Impossible speed
    track_b = {
        "track_id": "TRACK-0002",
        "camera_id": "CAM-002",
        "first_seen": now - 9,
        "last_seen": now,
        "body_embedding": [0.1] * 512,
        "clothing_upper": "#334455",
        "clothing_lower": "#112233"
    }
    link = tracker.correlate_tracks(track_a, track_b)
    assert link.is_physically_feasible is False
    assert link.overall_link_confidence < 0.30
