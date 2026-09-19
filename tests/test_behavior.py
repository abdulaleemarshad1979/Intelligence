"""Unit tests for CCTV behavioral analytics and suspicious activity detection."""

import pytest
import time
from app.behavior.activity_detector import CCTVBehaviorAnalyzer

def test_loitering_detection():
    analyzer = CCTVBehaviorAnalyzer(loiter_min_dwell_sec=10.0, loiter_max_displacement_px=50.0)
    now = time.time()
    # Trajectory of someone pacing within a 20px radius over 15 seconds
    history = []
    for i in range(20):
        t = now - 15.0 + (i * 0.75)
        # Circling around (300, 200)
        cx = 300 + 10 * (i % 2)
        cy = 200 + 8 * ((i + 1) % 2)
        history.append({
            "frame_id": i,
            "timestamp": t,
            "bbox": [cx - 20, cy - 40, 40, 80],
            "height_px": 80
        })

    alerts = analyzer.analyze_track_behavior("TRACK-TEST-1", "CAM-001", history)
    loiter_alerts = [a for a in alerts if a.alert_type == "LOITERING"]
    assert len(loiter_alerts) >= 1
    assert loiter_alerts[0].confidence >= 0.60

def test_sudden_sprint_detection():
    analyzer = CCTVBehaviorAnalyzer(sprint_velocity_threshold_px_per_sec=150.0)
    now = time.time()
    # Start walking slowly, then sudden fast surge
    history = []
    # Walking: 5 frames, 10px per sec
    for i in range(5):
        t = now - 4.0 + (i * 0.4)
        history.append({
            "frame_id": i,
            "timestamp": t,
            "bbox": [100 + i * 4, 200, 30, 80]
        })
    # Sprinting: jump 200px in 0.4s (500 px/s)
    for i in range(5, 10):
        t = now - 2.0 + ((i - 5) * 0.4)
        history.append({
            "frame_id": i,
            "timestamp": t,
            "bbox": [200 + (i - 5) * 120, 200, 30, 80]
        })

    alerts = analyzer.analyze_track_behavior("TRACK-TEST-2", "CAM-001", history)
    sprint_alerts = [a for a in alerts if a.alert_type == "SUDDEN_SPRINTING"]
    assert len(sprint_alerts) >= 1
    assert sprint_alerts[0].severity == "CRITICAL"
