# CCTV Intelligence Core: Multi-Modal Person Re-Identification & Behavioral Analytics Platform

A city-wide, CCTV-dedicated surveillance intelligence platform designed for police and investigative agencies. Rather than relying on single-frame face recognition (which fails when suspects turn away or wear masks/helmets), this platform fuses multi-frame temporal evidence across four independent modalities:
1. **3-Tier Partial Face Decomposition** (Upper/Mid/Lower tiers with mask/occlusion resilience).
2. **Body Proportions & Anatomical Lengths** (Torso length, leg length, torso-to-leg ratio, shoulder width, clothing color signatures).
3. **Calibrated Physical Stature** (Ground plane and camera tilt perspective projection).
4. **Walking Style & Gait Dynamics** (14 skeletal joints, inter-ankle stride waveform, cadence, pelvic bounce, posture correctness & spine tilt).

---

## Architecture Overview

```
                      CITY CCTV NETWORK (CAM-001 ... CAM-N)
                                     │
                                     ▼
                        PERSON DETECTION & TRACKING
                        (HOG / Motion Seg + Centroid Tracker)
                                     │
                                     ▼
                          MULTI-FRAME EVIDENCE BUFFER
                          (Best Quality Frame Selection)
                                     │
                 ┌───────────────────┼───────────────────┐
                 ▼                   ▼                   ▼
          3-TIER FACE          BODY BIOMETRICS      GAIT DYNAMICS
         DECOMPOSITION        & PROPORTIONS       & POSTURE LEAN
         - Upper (0-33%)      - Torso/Leg Ratio   - Stride Wave S(t)
         - Mid   (33-66%)     - Clothing Palette  - Cadence (steps/s)
         - Lower (66-100%)    - Calibrated Height - Posture Correctness
                 │                   │                   │
                 └───────────────────┼───────────────────┘
                                     ▼
                      DYNAMIC EVIDENCE FUSION ENGINE
                                     │
                ┌────────────────────┴────────────────────┐
                ▼                                         ▼
         FACE IS AVAILABLE                      FACE MASKED / REAR VIEW
      40% Face + 25% Body +                  40% Body + 45% Gait +
      25% Gait + 10% Height                  15% Height (Face = 0%)
                │                                         │
                └────────────────────┬────────────────────┘
                                     ▼
                        CRIMINAL GALLERY RETRIEVAL
                        (East Godavari Master FIR Database)
                                     │
                                     ▼
                    RANKED CANDIDATE MATCHES FOR REVIEW
                    - High Confidence (>= 78%)
                    - Review Required (52% - 78%)
                    - Unknown Person  (< 52%)
```

---

## Production Framework Governance & Model Recommendations

| Component | Recommended Framework | License | Production Status & Rationale |
| :--- | :--- | :--- | :--- |
| **Object Detection** | RT-DETR | Apache 2.0 | **Approved.** Avoids AGPL-3.0 copyleft exposure associated with YOLOv8. Native ONNX/TensorRT support. |
| **Pose Estimation** | RTMPose (MMPose) | Apache 2.0 | **Approved.** Sub-millisecond latency on edge nodes, robust occluded keypoint recovery, license-clean. |
| **Gait Signature** | Handcrafted Kinematics | Proprietary IP | **Approved.** Derived from joint angle velocities, stride frequency, and FFT harmonic ratios. Zero 3rd-party licensing risk. |
| **Deep Gait Models** | OpenGait (GaitSet / DeepGait) | Research / Proprietary | **Quarantine.** Avoid bundling into production builds due to commercial license restrictions and ties to proprietary overseas vendor codebases. |

---

## 3. Signal Fusion Architecture & Disparity Veto

At 1:N scale across an entire city or district gallery, soft biometrics alone yield unacceptably high false-match rates. Soft signals must act as conditional confirmations or hard geometric pruning gates rather than independent identity verifiers.

### Architectural Rules:
1. **Hard Geometric Pruning Gates**:
   - **Height Disparity Veto**: Any candidate with stature disparity $> 12$ cm is immediately pruned before gallery ranking ($H_{delta} > 12.0\text{ cm}$).
   - **Proportion Disparity Veto**: Torso-to-leg ratio disparity $> 0.30$ triggers an automatic geometric veto.
   - **Kinematic Stride Veto**: Incompatible stride length delta $> 25$ cm prunes candidate from matches.

2. **Conditional Confirmation Constraint**:
   - Soft biometrics (gait waveforms, calibrated stature, clothing signatures, body proportions) act as **conditional confirmations** when a primary hard biometric (frontal/partial face) is confirmed.
   - When primary facial biometrics are occluded, masked, or rear-facing, soft biometrics can **never** independently verify identity or trigger automatic high-confidence alerts ($C_{soft\_only} \le 0.45$). They route exclusively to the **Officer Review Gate** for mandatory human inspection.

---

## Directory Structure

```
cctv-intelligence/
├── app/
│   ├── main.py                   # FastAPI REST and live MJPEG streaming server
│   ├── detection/
│   │   └── person_detector.py    # Pedestrian detector with NMS
│   ├── tracking/
│   │   ├── tracker.py            # Multi-person tracker with persistent IDs
│   │   └── track_manager.py      # Multi-frame buffer & best quality frame selector
│   ├── features/
│   │   ├── face.py               # 3-Tier Face partition (Upper/Mid/Lower)
│   │   ├── partial_face.py       # Mask resilience & partial face matching
│   │   ├── body.py               # Body length, torso/leg ratio, clothing colors
│   │   ├── height.py             # Perspective calibrated physical stature (cm)
│   │   ├── pose.py               # 14 skeletal joint estimator
│   │   └── gait.py               # Stride waveform, cadence, posture correctness
│   ├── reid/
│   │   ├── embedding.py          # Vector similarity metrics
│   │   ├── gallery.py            # Suspect gallery manager
│   │   └── matcher.py            # Candidate matcher
│   ├── fusion/
│   │   └── evidence.py           # Multi-criteria evidence fusion engine
│   └── database/
│       ├── database.py           # SQLite database connection & schema
│       ├── models.py             # Dataclass data models
│       └── repository.py         # DB CRUD layer
├── config/
│   ├── config.yaml               # System weights, thresholds, video settings
│   └── cameras.yaml              # Camera topology & calibration parameters
├── data/
│   ├── samples/                  # Sample CCTV video
│   ├── tracks/                   # Extracted evidence crops
│   └── police_records.db         # Seeded suspect database from master_fir.csv
├── frontend/
│   ├── index.html                # Unified Command Center Dashboard
│   └── assets/
│       ├── css/style.css         # Dark cyber law-enforcement UI theme
│       └── js/app.js             # Live stream canvas & suspect inspection logic
├── scripts/
│   ├── seed_database.py          # Seeds criminal gallery from master_fir.csv
│   └── run_pipeline.py           # Batch & offline video analysis runner
├── tests/
│   ├── test_features.py          # Feature extraction unit tests
│   └── test_fusion.py            # Evidence fusion unit tests
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Installation

```bash
cd cctv-intelligence
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Seed Criminal Records Database

Import suspect profiles and biometrics from `master_fir.csv`:
```bash
python scripts/seed_database.py
```

### 3. Run Offline Batch Video Pipeline

Process the CCTV footage (`WhatsApp Video 2026-09-17 at 4.40.40 PM.mp4`):
```bash
python scripts/run_pipeline.py --video data/samples/cctv_sample.mp4 --max-frames 150 --stride 2
```

### 4. Launch Interactive Command Center Dashboard

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at `http://localhost:8000` to access the live dashboard.

---

## 🐳 Docker Deployment (One-Command)

Run the entire surveillance platform and PostgreSQL 16 database with Docker Compose:

```bash
# Build and launch all services (PostgreSQL 16 + CCTV Intelligence Core)
docker compose up -d --build

# View real-time logs
docker compose logs -f

# Check health status
docker compose ps

# Stop services
docker compose down
```

The system will be immediately accessible at:
- **Web Command Center & Live CCTV Feeds:** `http://localhost:8000`
- **PostgreSQL 16 Biometric Database:** `localhost:5432` (`cctv_intelligence`)
- **Watchlist Pipeline Status Endpoint:** `http://localhost:8000/api/watchlist/status`
- **Target Person Search API:** `http://localhost:8000/api/target-search`

---

## 🎯 Multi-Camera Target Person Search

The platform supports fleet-wide asynchronous target person search across multiple live CCTV cameras without duplicating RTSP pipelines or blocking live video streaming.

### Operational Workflow

```
Operator uploads reference image of person (person.jpg)
                    ↓
        Create target-search session (session_id)
                    ↓
     Select or monitor multiple cameras (CAM-001 ... CAM-N or *)
                    ↓
        Read existing live RTSP streams (CameraStreamManager)
                    ↓
          Detect people continuously (YOLO / RT-DETR)
                    ↓
             Track each person (ByteTrack / BoT-SORT)
                    ↓
      Compare observed person to reference (Face / OSNet Re-ID)
                    ↓
       Aggregate multiple observations (Rolling buffer)
                    ↓
       Confirm candidate across frames (Temporal confirmation)
                    ↓
       Capture best evidence image (Quality-weighted selection)
                    ↓
 Display camera + time + track + evidence (Dashboard & API)
```

### Key Architectural Guarantees

1. **Non-Blocking Asynchronous Pipeline**: Video streaming and ingestion run independently on dedicated grabber threads. Target AI inference runs asynchronously without stalling client MJPEG feeds.
2. **Track-Level Temporal Confirmation**: Never trusts a single frame. A candidate only transitions to `CONFIRMED_CANDIDATE` after meeting configurable temporal rules:
   - `min_confirmations: 4`
   - `confirmation_window_sec: 8`
   - `minimum_raw_similarity: 0.55`
   - `minimum_quality_score: 0.35`
   - `duplicate_event_cooldown_sec: 15`
3. **No Artificial Confidence Mapping**: Replaced artificial 68–98% percentage mapping with genuine measurements:
   - `raw_similarity`
   - `quality_score`
   - `detection_score`
   - `confirmation_count`
   - `track_consistency`
   - `confirmation_score = 0.50 * mean_recent_sim + 0.20 * max_sim + 0.15 * mean_quality + 0.15 * track_consistency`
4. **Best Evidence Frame Capture**: Selects the highest quality observation frame using:
   `best_frame_score = 0.45 * image_quality + 0.30 * raw_similarity + 0.15 * detection_score + 0.10 * crop_area_score`
   Saves full scene frame, zoomed person crop, face crop, `metadata.json`, and `hashes.json` with SHA-256 cryptographic signatures.
5. **Cross-Camera Correlation**: Correlates candidate appearances across cameras (`CAM-001 → CAM-004 → CAM-007`) using spatio-temporal travel feasibility without collapsing identity confirmation into cross-camera tracking.

### REST API Endpoints

- `POST /api/target-search/start`: Start target search session (multipart: image, name, mode, cameras).
- `GET /api/target-search/{session_id}`: Runtime status, active cameras count, candidate counts.
- `GET /api/target-search/{session_id}/events`: List of confirmed candidate events.
- `GET /api/target-search/{session_id}/cameras`: Real-time camera connection and active track telemetry.
- `GET /api/target-search/{session_id}/evidence`: Cryptographic evidence artifacts and SHA-256 digests.
- `POST /api/target-search/{session_id}/stop`: Terminate target search without restarting CCTV streams.


