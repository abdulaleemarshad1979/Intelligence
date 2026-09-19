"""Demo Seed Script for Gotham-Style Investigation Scenario.

Seeds the exact incident described in the mission specification:
INCIDENT #INC-2026-0041 at CAM-017 (18:42:11)
Candidate Observations:
- 18:42 CAM-017 Track 481
- 18:49 CAM-018 Track 774
- 18:56 CAM-021 Track 991
- 19:05 CAM-023 Track 144
"""

import time
import numpy as np
from app.database.repository import Repository
from app.database.models import (
    IncidentCase, PersonTarget, TrackObservation,
    ObservationRecord, FeatureRecord, CameraEntity
)

def generate_deterministic_vector(seed: int, dim: int = 512) -> list:
    np.random.seed(seed)
    v = np.random.randn(dim).astype(np.float32)
    v = v / np.linalg.norm(v)
    return v.tolist()

def seed_gotham_demo(repo: Repository) -> IncidentCase:
    """Seed the reference Gotham investigation scenario into the database."""
    base_time = 1774012931.0  # Represents 18:42:11

    # Base reference body embedding
    base_body_vec = generate_deterministic_vector(42, 512)
    base_gait_wave = [float(np.sin(i * 0.4) * 0.8) for i in range(32)]

    # 1. Cameras
    cameras = [
        CameraEntity("CAM-017", "Sector 4 Commercial Plaza North Exit", 16.9920, 82.2450, "Central Division", "NORTH"),
        CameraEntity("CAM-018", "North Transit Avenue Checkpoint", 16.9958, 82.2455, "Central Division", "NORTH"),
        CameraEntity("CAM-021", "Metro Junction North Underpass", 16.9998, 82.2460, "Central Division", "NORTH"),
        CameraEntity("CAM-023", "Ring Road North Terminal Perimeter", 17.0045, 82.2468, "North Division", "NORTH")
    ]
    for cam in cameras:
        repo.save_camera(cam)

    # 2. Incident
    incident = IncidentCase(
        incident_id="INC-2026-0041",
        case_number="INC-2026-0041",
        title="Commercial Corridor Break-In / Suspect Transit",
        description="Suspect spotted exiting Sector 4 Plaza heading northward. Wore face covering and carried dark backpack.",
        camera_id="CAM-017",
        incident_time=base_time,
        status="OPEN",
        priority="CRITICAL",
        officer_in_charge="AP-EG-8821",
        seed_track_id="481",
        created_at=time.time()
    )
    repo.save_incident(incident)

    # 3. Person of Interest
    poi = PersonTarget(
        person_id="POI-2026-0041",
        target_code="TARGET-2026-0041",
        canonical_name="UNIDENTIFIED_MALE_TARGET",
        status="PERSON_OF_INTEREST",
        notes="Suspect associated with Case #INC-2026-0041; northbound flight path across sector cameras.",
        created_at=time.time(),
        updated_at=time.time()
    )
    repo.save_person(poi)

    # 4. Tracks, Observations, Features
    # Track 481: 18:42:11 at CAM-017
    # Track 774: 18:49:05 at CAM-018 (+414s, 420m)
    # Track 991: 18:56:30 at CAM-021 (+445s, 450m)
    # Track 144: 19:05:14 at CAM-023 (+524s, 520m)
    scenario_tracks = [
        {
            "track_id": "481",
            "camera_id": "CAM-017",
            "t_offset": 0.0,
            "face_status": "UNAVAILABLE",
            "clothing_upper": "#1b2430",
            "clothing_lower": "#2c3539",
            "height_cm": 178.0,
            "noise_scale": 0.0,
            "carried": ["backpack"],
            "direction": "NORTH"
        },
        {
            "track_id": "774",
            "camera_id": "CAM-018",
            "t_offset": 414.0,  # 6m 54s later (18:49:05)
            "face_status": "UNAVAILABLE",
            "clothing_upper": "#1b2430",
            "clothing_lower": "#2c3539",
            "height_cm": 177.6,
            "noise_scale": 0.04,
            "carried": ["backpack"],
            "direction": "NORTH"
        },
        {
            "track_id": "991",
            "camera_id": "CAM-021",
            "t_offset": 859.0,  # 14m 19s later (18:56:30)
            "face_status": "UNAVAILABLE",
            "clothing_upper": "#1d2633",
            "clothing_lower": "#2a3337",
            "height_cm": 178.4,
            "noise_scale": 0.05,
            "carried": ["backpack"],
            "direction": "NORTH"
        },
        {
            "track_id": "144",
            "camera_id": "CAM-023",
            "t_offset": 1383.0,  # 23m 03s later (19:05:14)
            "face_status": "PARTIAL_UPPER",
            "clothing_upper": "#1b2430",
            "clothing_lower": "#2c3539",
            "height_cm": 178.0,
            "noise_scale": 0.045,
            "carried": ["backpack"],
            "direction": "NORTH"
        }
    ]

    for item in scenario_tracks:
        tid = item["track_id"]
        cid = item["camera_id"]
        t_stamp = base_time + item["t_offset"]

        # Vector with slight natural sensor variance
        np.random.seed(int(tid))
        noise = np.random.randn(512).astype(np.float32) * item["noise_scale"]
        b_vec = np.array(base_body_vec, dtype=np.float32) + noise
        b_vec = (b_vec / np.linalg.norm(b_vec)).tolist()

        # 1. Track Observation in tracks table
        t_obs = TrackObservation(
            track_id=tid,
            camera_id=cid,
            first_seen=t_stamp,
            last_seen=t_stamp + 25.0,
            frame_count=75,
            best_frame_path="/frontend/assets/placeholder.jpg",
            face_visible=False,
            face_status=item["face_status"],
            estimated_height_cm=item["height_cm"],
            clothing_upper=item["clothing_upper"],
            clothing_lower=item["clothing_lower"],
            stride_length_cm=68.0,
            cadence_steps_per_sec=1.8,
            spine_tilt_deg=4.2,
            gait_wave=base_gait_wave,
            body_embedding=b_vec,
            gait_embedding=base_gait_wave
        )
        repo.save_track(t_obs)

        # 2. Observation record
        obs = ObservationRecord(
            observation_id=f"OBS-{cid}-{tid}",
            track_id=tid,
            camera_id=cid,
            timestamp=t_stamp,
            frame_number=120,
            crop_path="/frontend/assets/placeholder.jpg",
            bbox={"x": 340, "y": 180, "w": 120, "h": 280}
        )
        repo.save_observation(obs)

        # 3. Feature record
        feat = FeatureRecord(
            feature_id=f"FEAT-{tid}",
            track_id=tid,
            observation_id=obs.observation_id,
            face_status=item["face_status"],
            face_embedding=[] if item["face_status"] == "UNAVAILABLE" else generate_deterministic_vector(99, 128),
            body_embedding=b_vec,
            gait_embedding=base_gait_wave,
            pose_landmarks={"cadence": 1.8, "spine_tilt": 4.2},
            clothing_attributes={"upper_color": item["clothing_upper"], "lower_color": item["clothing_lower"]},
            height_cm=item["height_cm"],
            carried_objects=item["carried"],
            direction=item["direction"],
            created_at=time.time()
        )
        repo.save_feature(feat)

    repo.log_audit(
        action="GOTHAM_DEMO_SEEDED",
        details="Seeded incident #INC-2026-0041 with CAM-017, CAM-018, CAM-021, and CAM-023 tracks",
        operator="SYSTEM"
    )

    return incident
