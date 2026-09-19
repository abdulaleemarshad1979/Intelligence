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
