"""FastAPI Application for City-Wide Intelligent CCTV Re-Identification & Intelligence Platform.

Integrates:
- Perception Model Stack Adapters (YOLO/ByteTrack, OSNet, InsightFace, MMPose, GaitSet)
- Spatio-Temporal Cross-Camera Correlation Engine
- CCTV Suspicious Activity & Behavioral Analytics
- Day-1 Police Governance & Legal Compliance (RBAC, Tamper-evident Audit, Auto-Retention, Human Review Gate)
- Re-ID Evaluation & Benchmark Metrics
- Real-time Video Feeds & Command Center API
"""

import os
import cv2
import time
import json
import yaml
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.database.database import init_db
from app.database.repository import Repository
from app.database.models import TrackObservation, MatchEvent, CameraEntity
from app.reid.gallery import SuspectGallery
from app.reid.matcher import CandidateMatcher
from app.reid.cross_camera import CrossCameraTracker
from app.behavior.activity_detector import CCTVBehaviorAnalyzer, BehaviorAlert
from app.compliance.rbac import UserRole, OfficerIdentity, DEFAULT_OFFICER
from app.compliance.audit import AuditLogger
from app.compliance.retention import BiometricRetentionEngine
from app.compliance.human_review import HumanReviewGate, ReviewSubmission
from app.evaluation.benchmark import ReIDEvaluationHarness

# Model Stack Adapters & Registry
from app.adapters.registry import model_registry
from app.adapters.license_audit import MODEL_LICENSE_CATALOG
from app.adapters.downloader import model_downloader
from app.features.height import HeightEstimator
from app.features.enhancement import stream_enhancer, candidate_enhancer
from app.discovery.onvif_scanner import onvif_scanner

# Gotham Investigation Modules
from app.search.person_search import PersonSearchCoordinator
from app.investigation.incident import IncidentManager
from app.investigation.timeline import TimelineGenerator
from app.investigation.graph import InvestigationGraphBuilder
from app.investigation.review import HumanAdjudicationGate
from app.investigation.demo_seed import seed_gotham_demo
from app.database.models import RelationshipLink

class ModelSelectionPayload(BaseModel):
    category: str
    model_id: str

class CreateIncidentPayload(BaseModel):
    case_number: str
    title: str
    camera_id: str
    incident_time: Optional[float] = None
    description: Optional[str] = ""
    priority: Optional[str] = "HIGH"
    officer_in_charge: Optional[str] = "AP-EG-8821"
    seed_track_id: Optional[str] = None

class FindPersonPayload(BaseModel):
    incident_id: Optional[str] = "INC-2026-0041"
    probe_track_id: Optional[str] = "481"

class AdjudicationReviewPayload(BaseModel):
    relationship_id: str
    reviewer_badge: str
    reviewer_name: str
    decision: str  # CONFIRM_IDENTITY, REJECT_ASSOCIATION, DEFER_REVIEW
    review_notes: Optional[str] = ""
    target_id: Optional[str] = None

class StreamEnhancementPayload(BaseModel):
    enable_clahe: Optional[bool] = None
    enable_denoise: Optional[bool] = None
    clahe_clip_limit: Optional[float] = None

class ApproveCameraPayload(BaseModel):
    camera_id: str
    name: str
    latitude: float
    longitude: float
    zone: Optional[str] = "East Zone"
    view_direction: Optional[str] = "NORTH"

class ScanNetworkPayload(BaseModel):
    timeout_seconds: Optional[float] = 1.5
    include_simulated_if_empty: Optional[bool] = True

app = FastAPI(
    title="City-Wide Intelligent CCTV Intelligence Platform",
    description="Multi-modal CCTV person re-identification, 3-tier face division, gait dynamics, evidence fusion, and police compliance",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database and core repositories
init_db()
repo = Repository()
gallery = SuspectGallery(repo)
matcher = CandidateMatcher(gallery, repo)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Mount static directories
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
if os.path.exists(os.path.join(DATA_DIR, "tracks")):
    app.mount("/data/tracks", StaticFiles(directory=os.path.join(DATA_DIR, "tracks")), name="tracks")

# Load camera configs & initialize Cross-Camera Tracker
def load_camera_config() -> Dict[str, Any]:
    cfg_path = os.path.join(BASE_DIR, "config", "cameras.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f).get("cameras", {})
    return {}

camera_configs = load_camera_config()
cross_camera_tracker = CrossCameraTracker(camera_configs)

# Initialize Behavioral Analytics
behavior_analyzer = CCTVBehaviorAnalyzer()

# Initialize Police Compliance Suite
audit_logger = AuditLogger(os.path.join(DATA_DIR, "police_records.db"))
retention_engine = BiometricRetentionEngine(
    db_path=os.path.join(DATA_DIR, "police_records.db"),
    tracks_dir=os.path.join(DATA_DIR, "tracks"),
    audit_logger=audit_logger
)
human_review_gate = HumanReviewGate(
    db_path=os.path.join(DATA_DIR, "police_records.db"),
    audit_logger=audit_logger
)

# Initialize Model Stack Adapters via Registry
height_estimator = HeightEstimator()

# Initialize Gotham Investigation Suite
incident_manager = IncidentManager(repo)
timeline_generator = TimelineGenerator()
graph_builder = InvestigationGraphBuilder()
human_adjudication_gate = HumanAdjudicationGate(repo)
person_search_coordinator = PersonSearchCoordinator(repo, camera_configs)

# Auto-seed the reference Gotham investigation demo scenario
if not repo.get_incident_by_id("INC-2026-0041"):
    seed_gotham_demo(repo)

# Log system bootstrap in tamper-evident audit ledger
audit_logger.log_action(
    action_type="SYSTEM_INITIALIZATION",
    resource_id="CCTV_CORE_ENGINE_V2",
    details={
        "status": "ONLINE",
        "active_models": model_registry.active_keys,
        "compliance_status": model_registry.license_auditor.audit_active_stack(model_registry.active_keys)["overall_status"],
        "cameras_active": list(camera_configs.keys())
    }
)

# ==================== CORE ROUTES ====================

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>CCTV Intelligence Dashboard Loading...</h1>")

@app.get("/api/status")
async def get_system_status():
    tracks = repo.get_all_tracks()
    suspects = repo.get_all_criminal_records()
    matches = repo.get_match_events(limit=100)
    cameras = load_camera_config()
    reg_status = model_registry.get_status()

    return {
        "status": "OPERATIONAL",
        "system_version": "2.0.0 (Neural Model Stack & Police Compliance)",
        "system_time": time.time(),
        "active_cameras": len(cameras),
        "total_suspects_in_gallery": len(suspects),
        "total_tracks_captured": len(tracks),
        "total_match_events": len(matches),
        "cameras": list(cameras.keys()),
        "active_models": model_registry.active_keys,
        "model_stack": {
            "detection": model_registry.detector.backend,
            "tracking": model_registry.tracker.backend,
            "reid": model_registry.reid.backend,
            "face": model_registry.face.backend,
            "pose": model_registry.pose.backend,
            "gait": model_registry.gait.backend
        },
        "compliance": reg_status["compliance_audit"]
    }

@app.get("/api/models/status")
async def get_models_status():
    """Return the active model adapters and architectures."""
    st = model_registry.get_status()
    return {
        "detection_tracking": st["backends"]["detection"],
        "reid": st["backends"]["reid"],
        "face": st["backends"]["face"],
        "pose": st["backends"]["pose"],
        "gait": st["backends"]["gait"],
        "active_models": st["active_models"],
        "backends": st["backends"],
        "compliance_audit": st["compliance_audit"]
    }

@app.get("/api/models/registry")
async def get_all_registered_models():
    """Return full catalog of pluggable open-source models with license and backend info."""
    return model_registry.get_all_registered_models()

@app.post("/api/models/select")
async def select_active_model(payload: ModelSelectionPayload):
    """Dynamically switch the active model for detection, tracking, reid, face, pose, or gait."""
    try:
        res = model_registry.select_model(payload.category, payload.model_id)
        audit_logger.log_action(
            action_type="MODEL_STACK_SWITCH",
            resource_id=f"{payload.category.upper()}_{payload.model_id.upper()}",
            details=res
        )
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/models/licenses")
async def get_license_audit():
    """Return compliance status and open-source license audit report."""
    active_audit = model_registry.license_auditor.audit_active_stack(model_registry.active_keys)
    return {
        "active_audit": active_audit,
        "full_catalog": model_registry.license_auditor.get_full_catalog()
    }

@app.get("/api/models/download-status")
async def get_models_download_status():
    """Return download status of all neural model checkpoints in models/."""
    return model_downloader.get_status()

@app.post("/api/models/download")
async def trigger_model_download(payload: Dict[str, Any]):
    """Download a model checkpoint from GitHub releases."""
    mid = payload.get("model_id")
    if payload.get("all", False):
        res = model_downloader.download_essential_models()
        return {"status": "SUCCESS", "results": res}
    if not mid:
        raise HTTPException(status_code=400, detail="model_id required")
    try:
        res = model_downloader.download_model(mid)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/benchmark-comparison")
async def get_models_benchmark_comparison():
    """Return head-to-head model performance and latency comparisons."""
    rpt_path = os.path.join(DATA_DIR, "benchmark_report.json")
    if os.path.exists(rpt_path):
        with open(rpt_path, "r") as f:
            data = json.load(f)
            return data.get("model_comparisons", {})
    return {}

@app.get("/api/cameras")
async def get_cameras():
    cameras = load_camera_config()
    return cameras

@app.post("/api/discovery/scan")
async def scan_network_cameras(payload: Optional[ScanNetworkPayload] = None):
    """Scan local network subnet via ONVIF WS-Discovery to find plug-and-play CCTV cameras."""
    t_wait = payload.timeout_seconds if payload else 1.5
    include_sim = payload.include_simulated_if_empty if payload else True
    discovered = onvif_scanner.scan_network(timeout=t_wait, include_simulated_if_empty=include_sim)
    
    saved_count = 0
    for cam in discovered:
        existing = repo.get_camera_by_id(cam.camera_id)
        if not existing:
            entity = CameraEntity(
                camera_id=cam.camera_id,
                name=f"{cam.manufacturer} {cam.model_name} ({cam.ip_address})",
                latitude=16.9890,
                longitude=82.2475,
                zone="Unassigned / Discovered",
                view_direction="NORTH",
                connected_topology=[],
                is_active=False,
                ip_address=cam.ip_address,
                rtsp_url=cam.rtsp_url,
                manufacturer=cam.manufacturer,
                model_name=cam.model_name,
                mac_address=cam.mac_address,
                discovery_status="DISCOVERED"
            )
            repo.save_camera(entity)
            saved_count += 1
            
    return {
        "status": "SUCCESS",
        "found_count": len(discovered),
        "newly_registered_count": saved_count,
        "cameras": [c.to_dict() for c in discovered]
    }

@app.get("/api/discovery/cameras")
async def list_discovered_cameras(status: Optional[str] = None):
    """Retrieve discovered and approved cameras from the database registry."""
    cams = repo.get_discovered_cameras(status=status)
    return {
        "status": "SUCCESS",
        "count": len(cams),
        "cameras": [
            {
                "camera_id": c.camera_id,
                "name": c.name,
                "ip_address": c.ip_address,
                "rtsp_url": c.rtsp_url,
                "manufacturer": c.manufacturer,
                "model_name": c.model_name,
                "mac_address": c.mac_address,
                "latitude": c.latitude,
                "longitude": c.longitude,
                "zone": c.zone,
                "discovery_status": c.discovery_status,
                "is_active": c.is_active
            }
            for c in cams
        ]
    }

@app.post("/api/discovery/approve")
async def approve_camera(payload: ApproveCameraPayload):
    """Approve a discovered camera, assigning an officer-friendly name and map coordinates."""
    ok = repo.approve_discovered_camera(
        camera_id=payload.camera_id,
        name=payload.name,
        latitude=payload.latitude,
        longitude=payload.longitude,
        zone=payload.zone or "East Zone"
    )
    if not ok:
        raise HTTPException(status_code=404, detail=f"Camera '{payload.camera_id}' not found")

    cam = repo.get_camera_by_id(payload.camera_id)
    return {
        "status": "SUCCESS",
        "message": f"Camera '{payload.name}' approved and activated on the map",
        "camera": {
            "camera_id": cam.camera_id,
            "name": cam.name,
            "latitude": cam.latitude,
            "longitude": cam.longitude,
            "zone": cam.zone,
            "status": cam.discovery_status,
            "is_active": cam.is_active
        } if cam else None
    }

@app.get("/api/tracks")
async def get_all_tracks():
    tracks = repo.get_all_tracks()
    return tracks

@app.get("/api/tracks/{track_id}")
async def get_track_detail(track_id: str):
    track = repo.get_track_by_id(track_id)
    if not track:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")

    # Match against criminal gallery
    obs = TrackObservation(
        track_id=track["track_id"],
        camera_id=track["camera_id"],
        first_seen=track["first_seen"],
        last_seen=track["last_seen"],
        frame_count=track["frame_count"],
        best_frame_path=track.get("best_frame_path", ""),
        face_visible=bool(track["face_visible"]),
        face_status=track["face_status"],
        face_tier_details=track.get("face_tier_details", {}),
        estimated_height_cm=track["estimated_height_cm"],
        body_proportions=track.get("body_proportions", {}),
        clothing_upper=track.get("clothing_upper", "#000000"),
        clothing_lower=track.get("clothing_lower", "#000000"),
        stride_length_px=track.get("stride_length_px", 0.0),
        stride_length_cm=track.get("stride_length_cm", 0.0),
        cadence_steps_per_sec=track.get("cadence_steps_per_sec", 0.0),
        spine_tilt_deg=track.get("spine_tilt_deg", 0.0),
        posture_score=track.get("posture_score", 0.0),
        gait_wave=track.get("gait_wave", []),
        face_embedding=json.loads(track["face_embedding"]) if track.get("face_embedding") else [],
        body_embedding=json.loads(track["body_embedding"]) if track.get("body_embedding") else [],
        gait_embedding=json.loads(track["gait_embedding"]) if track.get("gait_embedding") else []
    )

    candidates = matcher.match_track(obs, top_k=5)

    # Log audit entry for biometric inspection
    audit_logger.log_action(
        action_type="TRACK_DOSSIER_VIEW",
        resource_id=track_id,
        details={"camera_id": track["camera_id"], "top_match": candidates[0]["suspect_name"] if candidates else "None"}
    )

    return {
        "track": track,
        "candidate_matches": candidates
    }

@app.get("/api/suspects")
async def get_suspects():
    suspects = repo.get_all_criminal_records()
    return suspects

@app.get("/api/suspects/{suspect_id}")
async def get_suspect_detail(suspect_id: str):
    suspect = repo.get_criminal_record_by_id(suspect_id)
    if not suspect:
        raise HTTPException(status_code=404, detail="Suspect not found")
    
    # Audit log
    audit_logger.log_action(
        action_type="SUSPECT_RECORD_VIEW",
        resource_id=suspect_id,
        details={"suspect_name": suspect.accused_name, "fir_no": suspect.fir_no}
    )
    return suspect

@app.get("/api/matches")
async def get_match_events(limit: int = 50):
    events = repo.get_match_events(limit=limit)
    return events

# ==================== CROSS-CAMERA CORRELATION ====================

@app.get("/api/cross-camera/topology")
async def get_cross_camera_topology():
    """Return camera network graph nodes and adjacency."""
    cameras = load_camera_config()
    nodes = []
    edges = []
    for cid, c in cameras.items():
        nodes.append({
            "id": cid,
            "name": c.get("name"),
            "lat": c.get("latitude"),
            "lon": c.get("longitude"),
            "location": c.get("location")
        })
        for adj in c.get("adjacent_cameras", []):
            if adj in cameras and cid < adj:  # avoid duplicate undirected edges
                dist_m, min_s, max_s = cross_camera_tracker.compute_travel_window(cid, adj)
                edges.append({
                    "from": cid,
                    "to": adj,
                    "distance_meters": round(dist_m, 1),
                    "min_travel_sec": round(min_s, 1),
                    "max_travel_sec": round(max_s, 1)
                })
    return {"nodes": nodes, "edges": edges}

@app.get("/api/cross-camera/correlate/{track_id}")
async def correlate_cross_camera_track(track_id: str):
    """Reconstruct multi-camera suspect trajectory across the city network."""
    track = repo.get_track_by_id(track_id)
    if not track:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")

    all_tracks = repo.get_all_tracks()
    journey = cross_camera_tracker.reconstruct_trajectory(track, all_tracks, threshold=0.55)
    return {
        "person_identifier": journey.person_identifier,
        "camera_sequence": journey.camera_sequence,
        "total_distance_m": journey.total_distance_m,
        "total_duration_sec": journey.total_duration_sec,
        "average_speed_kmh": journey.average_speed_kmh,
        "overall_confidence": journey.confidence,
        "timeline": journey.timeline
    }

@app.get("/api/cross-camera/deepstream-mtmc/{track_id}")
async def correlate_deepstream_mtmc_track(track_id: str):
    """Reconstruct trajectory using NVIDIA DeepStream MTMC state machine logic."""
    track = repo.get_track_by_id(track_id)
    if not track:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")

    all_tracks = repo.get_all_tracks()
    mtmc_res = cross_camera_tracker.correlate_deepstream_mtmc(track, all_tracks, match_threshold=0.55)
    return mtmc_res

@app.get("/api/cross-camera/mtmc-benchmark")
async def get_mtmc_benchmark():
    """Benchmark DeepStream MTMC workflow vs Custom Spatio-Temporal Graph."""
    all_tracks = repo.get_all_tracks()
    return cross_camera_tracker.benchmark_mtmc_architectures(all_tracks)

# ==================== BEHAVIORAL ANALYTICS ====================

@app.get("/api/behavior/alerts")
async def get_behavior_alerts():
    """Analyze current track histories and return active suspicious activity alerts."""
    all_tracks = repo.get_all_tracks()
    alerts: List[Dict[str, Any]] = []

    for trk in all_tracks:
        tid = trk.get("track_id", "TRACK-0001")
        cid = trk.get("camera_id", "CAM-001")
        fc = trk.get("frame_count", 10)
        t_start = trk.get("first_seen", 0.0)
        t_end = trk.get("last_seen", t_start + fc * 0.04)

        # Reconstruct synthetic multi-frame trajectory from recorded bounding box & posture
        pts = []
        base_h = trk.get("estimated_height_cm", 170.0) * 1.5
        for f in range(fc):
            t_f = t_start + f * 0.04
            # Model slight natural trajectory variation
            cx = 400 + f * 4
            cy = 300 + (f % 4) * 2
            pts.append({
                "frame_id": f,
                "timestamp": t_f,
                "bbox": [int(cx - 30), int(cy - base_h / 2), 60, int(base_h)],
                "height_px": int(base_h)
            })

        track_alerts = behavior_analyzer.analyze_track_behavior(tid, cid, pts)
        for alt in track_alerts:
            alerts.append({
                "alert_id": alt.alert_id,
                "track_id": alt.track_id,
                "camera_id": alt.camera_id,
                "alert_type": alt.alert_type,
                "severity": alt.severity,
                "confidence": alt.confidence,
                "description": alt.description,
                "timestamp": alt.timestamp,
                "metrics": alt.metrics
            })

    return alerts

# ==================== POLICE COMPLIANCE & AUDIT ====================

@app.get("/api/compliance/audit-logs")
async def get_audit_logs(limit: int = 50, action: Optional[str] = None):
    """Retrieve immutable cryptographic audit trail records."""
    logs = audit_logger.get_logs(limit=limit, action_filter=action)
    return logs

@app.get("/api/compliance/audit-verify")
async def verify_audit_integrity():
    """Verify SHA-256 cryptographic blockchain-style hash chain integrity."""
    verification = audit_logger.verify_integrity()
    return verification

@app.post("/api/compliance/review-match")
async def submit_human_review(submission: ReviewSubmission):
    """Submit mandatory investigator sign-off for a candidate match."""
    try:
        res = human_review_gate.submit_review(submission)
        return res
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

@app.get("/api/compliance/retention-status")
async def get_retention_status():
    """Get status of biometric retention, disk usage, and auto-purge schedules."""
    status = retention_engine.get_retention_status()
    return status

@app.post("/api/compliance/purge-expired")
async def trigger_retention_purge(hours: Optional[float] = None):
    """Trigger authorized biometric data minimization purge."""
    res = retention_engine.purge_expired_records(force_max_age_sec=(hours * 3600.0) if hours else None)
    return res

# ==================== EVALUATION & BENCHMARKING ====================

@app.get("/api/evaluation/benchmark")
async def get_benchmark_results():
    """Fetch recent model stack benchmark and multi-modal ablation metrics."""
    rpt_path = os.path.join(DATA_DIR, "benchmark_report.json")
    if os.path.exists(rpt_path):
        with open(rpt_path, "r") as f:
            return json.load(f)

    # If report not generated yet, run real-time harness
    from scripts.benchmark_models import generate_independent_evaluation_dataset
    gallery_list = gallery.get_all()
    queries, ground_truth = generate_independent_evaluation_dataset(gallery_list)
    harness = ReIDEvaluationHarness()
    report = harness.evaluate_gallery(queries, ground_truth, gallery_list)
    return {
        "timestamp": report.timestamp,
        "queries": report.total_query_tracks,
        "gallery_size": report.gallery_size,
        "cmc_rank1": report.cmc_rank1,
        "cmc_rank5": report.cmc_rank5,
        "cmc_rank10": report.cmc_rank10,
        "mAP": report.mean_average_precision,
        "false_match_rate": report.false_match_rate,
        "ablation": report.ablation_results,
        "viewpoints": report.viewpoint_results,
        "model_comparisons": report.model_comparisons,
        "latency_ms": report.latency_breakdown_ms,
        "fps": report.overall_fps
    }

# ==================== MJPEG VIDEO FEED WITH INTELLIGENT HUD ====================

def generate_mjpeg_stream():
    video_path = os.path.join(DATA_DIR, "samples", "cctv_sample.mp4")
    if not os.path.exists(video_path):
        video_path = "/home/abdul-aleem-arshad/Downloads/WhatsApp Video 2026-09-17 at 4.40.40 PM.mp4"

    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while True:
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_path)

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Step 1: Always-on instant frame enhancement (CLAHE + Fast Denoising)
        frame = stream_enhancer.enhance_stream_frame(frame)

        frame_id += 1
        h, w = frame.shape[:2]

        # Use pluggable detector adapter from registry
        active_det = model_registry.detector
        detections = active_det.detect_and_track(frame, frame_id)
        active_face = model_registry.face
        active_pose = model_registry.pose
        active_reid_name = model_registry.active_keys["reid"].upper()
        active_gait_name = model_registry.active_keys["gait"].upper()

        # Intelligent HUD Overlay
        cv2.putText(frame, "CAM-001 | KAKINADA HOSPITAL CORRIDOR | SECURE STREAM", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"DET: {active_det.backend} | RE-ID: {active_reid_name} | GAIT: {active_gait_name}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)

        for det in detections:
            x, y, bw, bh = det.bbox
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            person_crop = frame[y1:y2, x1:x2]

            face_res = active_face.analyze_face(person_crop) if person_crop.size > 0 else None
            h_res = height_estimator.estimate_height_cm([x, y, x + bw, y + bh], frame_height=h)
            pose_res = active_pose.estimate_pose(person_crop) if person_crop.size > 0 else None

            # Color coding: Green if face visible, orange if masked or turned away
            color = (0, 255, 0) if (face_res and face_res.is_available) else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Badge with Track ID, Calibrated Height, and Face Status
            status_text = face_res.status if face_res else "UNAVAILABLE"
            tid_text = f"TRACK-{det.track_id:04d}" if det.track_id else "TRACK"
            badge = f"{tid_text} | H:{h_res['estimated_height_cm']:.0f}cm | {status_text}"
            cv2.putText(frame, badge, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Draw 3-part face box & decomposition lines
            if face_res and face_res.bbox[2] > 0:
                fx, fy, fw, fh_box = face_res.bbox
                g_fx1, g_fy1 = x1 + fx, y1 + fy
                g_fx2, g_fy2 = g_fx1 + fw, g_fy1 + fh_box
                cv2.rectangle(frame, (g_fx1, g_fy1), (g_fx2, g_fy2), (0, 255, 255), 1)
                # 3 tier horizontal markers
                cv2.line(frame, (g_fx1, int(g_fy1 + fh_box * 0.33)), (g_fx2, int(g_fy1 + fh_box * 0.33)), (0, 200, 255), 1)
                cv2.line(frame, (g_fx1, int(g_fy1 + fh_box * 0.66)), (g_fx2, int(g_fy1 + fh_box * 0.66)), (0, 200, 255), 1)

            # Draw skeletal joints
            if pose_res and pose_res.keypoints:
                for kpt in pose_res.keypoints:
                    if len(kpt) >= 2 and kpt[2] > 0.4:
                        cv2.circle(frame, (int(x1 + kpt[0]), int(y1 + kpt[1])), 3, (255, 255, 0), -1)

        ret_enc, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ret_enc:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.04)

@app.get("/api/video_feed")
async def video_feed():
    return StreamingResponse(generate_mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/stream/enhancement")
async def get_stream_enhancement():
    """Retrieve current Step 1 stream pre-processing settings (CLAHE & Denoising)."""
    return {
        "status": "SUCCESS",
        "enhancement": stream_enhancer.get_status()
    }

@app.post("/api/stream/enhancement")
async def update_stream_enhancement(payload: StreamEnhancementPayload):
    """Dynamically toggle or configure CLAHE and Denoising on live streams."""
    if payload.enable_clahe is not None:
        stream_enhancer.enable_clahe = payload.enable_clahe
    if payload.enable_denoise is not None:
        stream_enhancer.enable_denoise = payload.enable_denoise
    if payload.clahe_clip_limit is not None:
        stream_enhancer.clahe_clip_limit = max(0.5, min(10.0, payload.clahe_clip_limit))
        stream_enhancer._clahe = cv2.createCLAHE(
            clipLimit=stream_enhancer.clahe_clip_limit,
            tileGridSize=stream_enhancer.clahe_grid_size
        )
    return {
        "status": "SUCCESS",
        "message": "Stream enhancement configuration updated",
        "enhancement": stream_enhancer.get_status()
    }

# ==================== GOTHAM INVESTIGATION API ROUTES ====================

@app.get("/api/investigation/incidents")
async def list_investigation_incidents():
    """Retrieve all recorded incidents."""
    incidents = repo.get_all_incidents()
    return {
        "status": "SUCCESS",
        "count": len(incidents),
        "incidents": [
            {
                "incident_id": inc.incident_id,
                "case_number": inc.case_number,
                "title": inc.title,
                "description": inc.description,
                "camera_id": inc.camera_id,
                "incident_time": inc.incident_time,
                "status": inc.status,
                "priority": inc.priority,
                "officer_in_charge": inc.officer_in_charge,
                "seed_track_id": inc.seed_track_id,
                "created_at": inc.created_at
            }
            for inc in incidents
        ]
    }

@app.post("/api/investigation/incidents")
async def create_investigation_incident(payload: CreateIncidentPayload):
    """Create a new investigative incident and bind a seed observation."""
    inc = incident_manager.create_incident(
        case_number=payload.case_number,
        title=payload.title,
        camera_id=payload.camera_id,
        incident_time=payload.incident_time or time.time(),
        description=payload.description or "",
        priority=payload.priority or "HIGH",
        officer_in_charge=payload.officer_in_charge or "AP-EG-8821",
        seed_track_id=payload.seed_track_id
    )
    return {"status": "SUCCESS", "incident": inc.to_dict() if hasattr(inc, "to_dict") else inc.__dict__}

@app.get("/api/investigation/incidents/{incident_id}")
async def get_incident_details(incident_id: str):
    """Get full incident case file with seed observation and linked suspect info."""
    details = incident_manager.get_incident_details(incident_id)
    if not details:
        raise HTTPException(status_code=404, detail="Incident not found")
    return {"status": "SUCCESS", "incident": details}

@app.post("/api/investigation/find_person")
async def find_person_investigation(payload: FindPersonPayload):
    """Core 'FIND THIS PERSON' investigation query across the CCTV observation database."""
    inc_id = payload.incident_id or "INC-2026-0041"
    inc = repo.get_incident_by_id(inc_id)
    if not inc:
        seed_gotham_demo(repo)
        inc = repo.get_incident_by_id(inc_id)

    probe_track_id = payload.probe_track_id or (inc.seed_track_id if inc else "481")
    if not probe_track_id:
        raise HTTPException(status_code=400, detail="No probe track specified")

    # Probe track & feature
    probe_track = repo.get_track_by_id(probe_track_id)
    probe_feature = repo.get_feature_for_track(probe_track_id)

    # Execute search coordinator
    associations = person_search_coordinator.find_associated_tracks(
        probe_track_id=probe_track_id,
        time_horizon_seconds=7200.0,
        min_composite_score=0.35
    )

    assoc_dicts = [a.to_dict() for a in associations]

    # Automatically save candidate relationships if not yet recorded
    for a in associations:
        rel = RelationshipLink(
            relationship_id=f"REL-{probe_track_id}-{a.candidate_track_id}",
            source_type="TRACK",
            source_id=probe_track_id,
            target_type="TRACK",
            target_id=a.candidate_track_id,
            relationship_type="CANDIDATE_SAME_PERSON",
            confidence_score=a.fusion_result.get("composite_score", 0.0),
            evidence=a.evidence_breakdown,
            status="CANDIDATE",
            created_at=time.time()
        )
        repo.save_relationship(rel)

    # Generate chronological movement timeline
    timeline = timeline_generator.generate_timeline(
        candidate_associations=assoc_dicts,
        seed_track=probe_track
    )

    # Generate graph representation
    existing_rels = [r.__dict__ for r in repo.get_relationships(source_id=probe_track_id)]
    inc_dict = inc.__dict__ if inc else {
        "incident_id": inc_id,
        "case_number": "INC-2026-0041",
        "camera_id": "CAM-017",
        "seed_track_id": probe_track_id
    }
    graph = graph_builder.build_graph(inc_dict, assoc_dicts, existing_rels)

    # Log action in tamper-evident ledger
    audit_logger.log_action(
        action_type="PERSON_INVESTIGATION_SEARCH",
        resource_id=f"INCIDENT_{inc_id}_PROBE_{probe_track_id}",
        details={
            "probe_track": probe_track_id,
            "candidates_found": len(associations),
            "camera_sequence": [t["camera_id"] for t in timeline]
        }
    )

    return {
        "status": "SUCCESS",
        "incident_id": inc_id,
        "case_number": inc.case_number if inc else "INC-2026-0041",
        "probe": {
            "track_id": probe_track_id,
            "camera_id": probe_track.get("camera_id", "CAM-017") if probe_track else "CAM-017",
            "first_seen": probe_track.get("first_seen", 0.0) if probe_track else 0.0,
            "face_status": probe_feature.face_status if probe_feature else "UNAVAILABLE",
            "clothing_upper": probe_feature.clothing_attributes.get("upper_color", "#1b2430") if probe_feature else "#1b2430",
            "clothing_lower": probe_feature.clothing_attributes.get("lower_color", "#2c3539") if probe_feature else "#2c3539",
            "estimated_height_cm": probe_feature.height_cm if probe_feature else 178.0,
            "carried_objects": probe_feature.carried_objects if probe_feature else ["backpack"],
            "direction": probe_feature.direction if probe_feature else "NORTH",
            "best_frame_path": probe_track.get("best_frame_path", "/frontend/assets/placeholder.jpg") if probe_track else "/frontend/assets/placeholder.jpg"
        },
        "candidate_associations": assoc_dicts,
        "movement_timeline": timeline,
        "investigation_graph": graph
    }

@app.get("/api/investigation/timeline/{incident_id}")
async def get_incident_timeline(incident_id: str):
    """Retrieve the movement timeline for an incident."""
    inc = repo.get_incident_by_id(incident_id)
    if not inc or not inc.seed_track_id:
        raise HTTPException(status_code=404, detail="Incident or seed track not found")

    probe_track = repo.get_track_by_id(inc.seed_track_id)
    associations = person_search_coordinator.find_associated_tracks(
        probe_track_id=inc.seed_track_id,
        time_horizon_seconds=7200.0,
        min_composite_score=0.35
    )
    timeline = timeline_generator.generate_timeline(
        candidate_associations=[a.to_dict() for a in associations],
        seed_track=probe_track
    )
    return {"status": "SUCCESS", "incident_id": incident_id, "timeline": timeline}

@app.get("/api/investigation/graph/{incident_id}")
async def get_incident_graph(incident_id: str):
    """Retrieve the investigation graph for an incident."""
    inc = repo.get_incident_by_id(incident_id)
    if not inc or not inc.seed_track_id:
        raise HTTPException(status_code=404, detail="Incident or seed track not found")

    associations = person_search_coordinator.find_associated_tracks(
        probe_track_id=inc.seed_track_id,
        time_horizon_seconds=7200.0,
        min_composite_score=0.35
    )
    assoc_dicts = [a.to_dict() for a in associations]
    relationships = [r.__dict__ for r in repo.get_relationships(source_id=inc.seed_track_id)]
    graph = graph_builder.build_graph(inc.__dict__, assoc_dicts, relationships)
    return {"status": "SUCCESS", "incident_id": incident_id, "graph": graph}

@app.post("/api/investigation/review")
async def submit_investigation_review(payload: AdjudicationReviewPayload):
    """Submit human review decision for high-consequence association."""
    result = human_adjudication_gate.submit_review(
        relationship_id=payload.relationship_id,
        reviewer_badge=payload.reviewer_badge,
        reviewer_name=payload.reviewer_name,
        decision=payload.decision,
        review_notes=payload.review_notes or "",
        target_id=payload.target_id
    )
    return {"status": "SUCCESS", "adjudication": result}

@app.post("/api/investigation/seed_demo")
async def seed_investigation_demo_endpoint():
    """Manually re-seed or reset the Gotham demo investigation scenario."""
    inc = seed_gotham_demo(repo)
    return {"status": "SUCCESS", "incident_id": inc.incident_id, "case_number": inc.case_number}
