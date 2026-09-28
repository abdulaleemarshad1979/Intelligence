"""Multi-Camera Target Person Search Coordinator.

Orchestrates multi-camera RTSP ingestion, decoupled AI perception,
track-level temporal confirmation, cross-camera correlation,
and evidence artifact generation.
"""

import os
import time
import uuid
import yaml
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
import cv2
import numpy as np

from app.target_search.models import (
    TargetSearchMode, CandidateStatus, SessionStatus,
    TargetSearchConfig, Observation, CandidateEvent, TargetSearchSession
)
from app.target_search.confirmation import TemporalConfirmationEngine
from app.vision.face_engine import FaceBiometricEngine
from app.adapters.registry import model_registry
from app.ingestion.stream_manager import CameraStreamManager, get_stream_manager
from app.database.repository import Repository
from app.reid.cross_camera import CrossCameraTracker
from app.reid.embedding import cosine_similarity

logger = logging.getLogger(__name__)


def load_target_search_config() -> TargetSearchConfig:
    """Load default target search parameters from config/config.yaml if present."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg_path = os.path.join(base_dir, "config", "config.yaml")
    defaults = TargetSearchConfig()

    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            ts_cfg = cfg.get("target_search", {})
            if "min_confirmations" in ts_cfg:
                defaults.min_confirmations = int(ts_cfg["min_confirmations"])
            if "confirmation_window_sec" in ts_cfg:
                defaults.confirmation_window_sec = float(ts_cfg["confirmation_window_sec"])
            if "minimum_raw_similarity" in ts_cfg:
                defaults.minimum_raw_similarity = float(ts_cfg["minimum_raw_similarity"])
            if "minimum_quality_score" in ts_cfg:
                defaults.minimum_quality_score = float(ts_cfg["minimum_quality_score"])
            if "duplicate_event_cooldown_sec" in ts_cfg:
                defaults.duplicate_event_cooldown_sec = float(ts_cfg["duplicate_event_cooldown_sec"])
        except Exception as ex:
            logger.debug(f"Failed to parse target_search config: {ex}")

    return defaults


class TargetSearchCoordinator:
    """Fleet-wide coordinator for target person search across multiple CCTV feeds."""

    def __init__(
        self,
        base_dir: Optional[str] = None,
        stream_manager: Optional[CameraStreamManager] = None,
        face_engine: Optional[FaceBiometricEngine] = None,
        repo: Optional[Repository] = None
    ):
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.storage_dir = os.path.join(self.data_dir, "target_search")
        os.makedirs(self.storage_dir, exist_ok=True)

        self.stream_manager = stream_manager or get_stream_manager(self.data_dir)
        self.face_engine = face_engine or FaceBiometricEngine()
        self.repo = repo or Repository()

        # Session registry: session_id -> TargetSearchSession
        self.sessions: Dict[str, TargetSearchSession] = {}
        # Confirmation engines: session_id -> TemporalConfirmationEngine
        self.engines: Dict[str, TemporalConfirmationEngine] = {}
        # Background worker threads: session_id -> Thread
        self._threads: Dict[str, threading.Thread] = {}
        self._stop_events: Dict[str, threading.Event] = {}

        self._lock = threading.Lock()

        # Initialize cross camera tracker topology
        self.cross_camera_tracker: Optional[CrossCameraTracker] = None
        self._init_cross_camera_tracker()

    def _init_cross_camera_tracker(self):
        """Build camera network topology for spatio-temporal transit feasibility."""
        try:
            cameras = self.repo.get_all_cameras()
            registry = {}
            for c in cameras:
                registry[c.camera_id] = {
                    "name": c.name,
                    "latitude": c.latitude,
                    "longitude": c.longitude,
                    "adjacent_cameras": [t.get("camera_id") for t in getattr(c, "connected_topology", []) if isinstance(t, dict)]
                }
            if registry:
                self.cross_camera_tracker = CrossCameraTracker(registry)
        except Exception as ex:
            logger.debug(f"CrossCameraTracker initialization note: {ex}")

    def decode_and_validate_reference(self, image_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """Decode image and validate minimum resolution and viability."""
        if image_input is None:
            raise ValueError("Reference image cannot be empty or unreadable.")

        img = None
        if isinstance(image_input, (bytes, bytearray)):
            if len(image_input) == 0:
                raise ValueError("Reference image cannot be empty or unreadable.")
            img = cv2.imdecode(np.frombuffer(image_input, np.uint8), cv2.IMREAD_COLOR)
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("Reference image cannot be empty or unreadable.")
            img = image_input
        elif isinstance(image_input, str):
            if not image_input.strip():
                raise ValueError("Reference image cannot be empty or unreadable.")
            if os.path.isfile(image_input):
                img = cv2.imread(image_input)
            elif image_input.startswith("data:image"):
                import base64
                header, encoded = image_input.split(",", 1)
                img = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
            else:
                img = None
        else:
            img = None

        if img is None or img.size == 0:
            raise ValueError("Reference image cannot be empty or unreadable.")

        h, w = img.shape[:2]
        if h < 24 or w < 24:
            raise ValueError(f"Reference image resolution ({w}x{h}) is too low for reliable matching.")

        return img

    def resolve_cameras(self, cameras_param: Union[str, List[str]]) -> List[str]:
        """Resolve requested camera IDs, expanding '*' to all available cameras."""
        all_cams = []
        try:
            db_cams = self.repo.get_all_cameras()
            all_cams = [c.camera_id for c in db_cams if getattr(c, "is_active", True)]
        except Exception:
            pass

        if not all_cams:
            all_cams = ["CAM-001", "CAM-002", "CAM-003", "CAM-004"]

        if isinstance(cameras_param, str):
            parts = [c.strip() for c in cameras_param.split(",") if c.strip()]
        else:
            parts = [str(c).strip() for c in cameras_param if str(c).strip()]

        if not parts or "*" in parts or "ALL" in [p.upper() for p in parts]:
            return all_cams

        # Deduplicate while preserving order
        resolved = []
        for p in parts:
            if p not in resolved:
                resolved.append(p)
        return resolved

    def start_search(
        self,
        image_input: Union[str, bytes, np.ndarray],
        name: Optional[str] = "Subject of Interest",
        mode: str = "auto",
        cameras: Union[str, List[str]] = "*",
        config: Optional[TargetSearchConfig] = None
    ) -> Dict[str, Any]:
        """Create target-search session and initiate asynchronous multi-camera tracking."""
        ref_img = self.decode_and_validate_reference(image_input)
        req_mode = str(mode).lower().strip()
        if req_mode not in ("auto", "face", "reid"):
            req_mode = "auto"

        resolved_cameras = self.resolve_cameras(cameras)
        if not resolved_cameras:
            raise ValueError("No valid cameras selected for target search.")

        # Generate unique session ID
        session_id = f"TS-{uuid.uuid4().hex[:8].upper()}"
        target_id = f"TGT-{session_id}"
        session_dir = os.path.join(self.storage_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Store reference image
        ref_path = os.path.join(session_dir, "reference.jpg")
        cv2.imwrite(ref_path, ref_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Extract reference biometric representations
        active_mode = req_mode
        ref_face_emb: Optional[List[float]] = None
        ref_reid_emb: Optional[List[float]] = None
        ref_face_bbox: Optional[List[int]] = None
        ref_quality: Dict[str, Any] = {}

        # 1. Try face extraction
        detected_faces = self.face_engine.detect_faces(ref_img)
        has_viable_face = False

        if detected_faces:
            best_face = max(detected_faces, key=lambda d: d.get("score", 0.0))
            bx, by, bw, bh = [int(v) for v in best_face["bbox"]]
            h_img, w_img = ref_img.shape[:2]
            bx, by = max(0, bx), max(0, by)
            bw, bh = min(bw, w_img - bx), min(bh, h_img - by)

            if bw >= 20 and bh >= 20:
                face_crop = ref_img[by:by + bh, bx:bx + bw]
                ref_quality = self.face_engine.assess_face_quality(face_crop)
                landmarks = best_face.get("landmarks")
                is_viable, emb = self.face_engine.extract_face_embedding(face_crop, landmarks, ref_img)
                if is_viable and np.linalg.norm(emb) > 1e-4:
                    ref_face_emb = [round(float(v), 6) for v in emb]
                    ref_face_bbox = [bx, by, bw, bh]
                    has_viable_face = True

        # 2. Extract Re-ID representation
        try:
            reid_adapter = model_registry.reid
            reid_res = reid_adapter.extract_embedding(ref_img)
            if reid_res and reid_res.embedding:
                ref_reid_emb = [round(float(v), 6) for v in reid_res.embedding]
        except Exception as r_ex:
            logger.debug(f"Re-ID feature extraction note: {r_ex}")

        # 3. Determine active operational mode
        if req_mode == "auto":
            if has_viable_face:
                active_mode = "face"
            elif ref_reid_emb is not None:
                active_mode = "reid"
            else:
                active_mode = "face"
        elif req_mode == "face":
            if not has_viable_face:
                raise ValueError("No viable face detected in reference image for required 'face' mode.")
            active_mode = "face"
        elif req_mode == "reid":
            if ref_reid_emb is None:
                raise ValueError("Could not extract body appearance Re-ID features from reference image.")
            active_mode = "reid"

        cfg = config or load_target_search_config()

        session = TargetSearchSession(
            session_id=session_id,
            target_id=target_id,
            name=name or "Subject of Interest",
            requested_mode=req_mode,
            active_mode=active_mode,
            selected_cameras=resolved_cameras,
            reference_image_path=ref_path,
            reference_face_embedding=ref_face_emb,
            reference_reid_embedding=ref_reid_emb,
            reference_face_bbox=ref_face_bbox,
            reference_quality=ref_quality,
            config=cfg,
            status=SessionStatus.RUNNING.value
        )

        confirmation_engine = TemporalConfirmationEngine(
            session_id=session_id,
            storage_base_dir=self.storage_dir,
            config=cfg
        )

        with self._lock:
            self.sessions[session_id] = session
            self.engines[session_id] = confirmation_engine
            stop_evt = threading.Event()
            self._stop_events[session_id] = stop_evt

        # Persist session to database
        try:
            self.repo.insert_target_search_session(session)
        except Exception as db_ex:
            logger.debug(f"DB session persistence note: {db_ex}")

        # Launch decoupled asynchronous AI pipeline thread
        thread = threading.Thread(
            target=self._session_pipeline_loop,
            args=(session_id, stop_evt),
            name=f"TargetSearch-{session_id}",
            daemon=True
        )
        self._threads[session_id] = thread
        thread.start()

        logger.info(
            f"Started Target Search [{session_id}] on {len(resolved_cameras)} cameras in '{active_mode}' mode."
        )

        return {
            "status": "SUCCESS",
            "session_id": session_id,
            "target_id": target_id,
            "selected_cameras": resolved_cameras
        }

    def _session_pipeline_loop(self, session_id: str, stop_event: threading.Event):
        """Asynchronous multi-camera AI search loop."""
        session = self.sessions.get(session_id)
        engine = self.engines.get(session_id)
        if not session or not engine:
            return

        last_frame_ids: Dict[str, int] = {}
        detector = model_registry.detector
        reid_model = model_registry.reid

        logger.info(f"[{session_id}] Asynchronous AI inference loop active.")

        while not stop_event.is_set():
            t_loop_start = time.perf_counter()

            for cam_id in session.selected_cameras:
                if stop_event.is_set():
                    break

                try:
                    worker = self.stream_manager.get_or_create_worker(cam_id)
                    # Safe non-blocking grab of latest frame
                    frame, frame_id, f_time = worker.get_latest_frame()

                    if frame is None or frame.size == 0:
                        continue

                    # Process each frame only once
                    if last_frame_ids.get(cam_id) == frame_id:
                        continue
                    last_frame_ids[cam_id] = frame_id

                    h, w = frame.shape[:2]
                    now = f_time or time.time()

                    # 1. Person Detection & Tracking
                    detections = detector.detect_and_track(frame, frame_id)
                    if not detections:
                        continue

                    for det in detections:
                        if stop_event.is_set():
                            break

                        track_id = getattr(det, "track_id", None)
                        if not track_id:
                            continue

                        bx, by, bw, bh = [int(v) for v in det.bbox]
                        bx1, by1 = max(0, bx), max(0, by)
                        bx2, by2 = min(w, bx + bw), min(h, by + bh)
                        if (bx2 - bx1) < 15 or (by2 - by1) < 25:
                            continue

                        person_crop = frame[by1:by2, bx1:bx2]
                        if person_crop.size == 0:
                            continue

                        raw_sim = 0.0
                        quality_score = 0.0
                        det_score = float(getattr(det, "confidence", 0.85))
                        face_crop_candidate = None
                        matched_face_bbox = None
                        sim_type = session.active_mode

                        # 2. Feature Extraction & Comparison
                        if session.active_mode == "face":
                            # Detect faces in upper portion of person crop or full frame
                            f_faces = self.face_engine.detect_faces(person_crop)
                            if f_faces and session.reference_face_embedding:
                                best_f = max(f_faces, key=lambda f: d_score if (d_score := f.get("score", 0.0)) else 0.0)
                                fx, fy, fw, fh = [int(v) for v in best_f["bbox"]]
                                if fw > 10 and fh > 10 and best_f.get("is_viable", True):
                                    face_crop_candidate = person_crop[fy:fy + fh, fx:fx + fw]
                                    matched_face_bbox = [bx1 + fx, by1 + fy, fw, fh]
                                    q_eval = best_f.get("quality", {})
                                    quality_score = float(q_eval.get("quality_score", 0.5))

                                    is_v, cand_emb = self.face_engine.extract_face_embedding(
                                        face_crop_candidate,
                                        best_f.get("landmarks"),
                                        person_crop
                                    )
                                    if is_v:
                                        raw_sim = self.face_engine.compute_face_similarity(
                                            cand_emb,
                                            session.reference_face_embedding
                                        )
                            elif session.requested_mode == "auto" and session.reference_reid_embedding:
                                # Fall back to Re-ID if face not visible in this frame
                                sim_type = "reid"
                                reid_res = reid_model.extract_embedding(person_crop)
                                quality_score = float(reid_res.quality_score)
                                raw_sim = cosine_similarity(reid_res.embedding, session.reference_reid_embedding)

                        elif session.active_mode == "reid":
                            if session.reference_reid_embedding:
                                reid_res = reid_model.extract_embedding(person_crop)
                                quality_score = float(reid_res.quality_score)
                                raw_sim = cosine_similarity(reid_res.embedding, session.reference_reid_embedding)

                        # Emit observation
                        obs = Observation(
                            timestamp=now,
                            camera_id=cam_id,
                            track_id=str(track_id),
                            bbox=[bx1, by1, bx2 - bx1, by2 - by1],
                            raw_similarity=float(raw_sim),
                            quality_score=float(quality_score),
                            detection_score=float(det_score),
                            frame_id=frame_id,
                            face_bbox=matched_face_bbox,
                            similarity_type=sim_type
                        )

                        track_state, event = engine.process_observation(
                            observation=obs,
                            full_frame=frame,
                            person_crop=person_crop,
                            face_crop=face_crop_candidate
                        )

                        # Handle newly confirmed candidate event
                        if event:
                            self._handle_confirmed_event(session, event)

                except Exception as cam_ex:
                    logger.debug(f"[{session_id}] Search error on camera {cam_id}: {cam_ex}")

            # Prune stale tracks occasionally
            engine.prune_expired_tracks(time.time())

            # Throttled sleep (~8-10 Hz inference cadence)
            elapsed = time.perf_counter() - t_loop_start
            sleep_duration = max(0.02, 0.10 - elapsed)
            time.sleep(sleep_duration)

        session.status = SessionStatus.STOPPED.value
        session.stopped_at = time.time()
        logger.info(f"[{session_id}] Target search pipeline terminated gracefully.")

    def _handle_confirmed_event(self, session: TargetSearchSession, event: CandidateEvent):
        """Cross-camera association and database persistence for confirmed events."""
        # 1. Evaluate cross-camera spatio-temporal link if prior events exist
        if self.cross_camera_tracker and len(session.selected_cameras) > 1:
            all_events = self.engines[session.session_id].get_events()
            prior_events = [e for e in all_events if e.event_id != event.event_id and e.camera_id != event.camera_id]
            if prior_events:
                # Find most recent prior sighting
                prior = max(prior_events, key=lambda e: e.last_seen)
                t_delta = event.first_seen - prior.last_seen
                dist_m, min_s, max_s = self.cross_camera_tracker.compute_travel_window(prior.camera_id, event.camera_id)
                feasible = (t_delta >= 0 and (dist_m <= 50.0 or t_delta >= min_s * 0.7))

                event.metadata["cross_camera_correlation"] = {
                    "previous_camera": prior.camera_id,
                    "previous_event": prior.event_id,
                    "time_delta_sec": round(t_delta, 1),
                    "distance_meters": round(dist_m, 1),
                    "is_feasible": feasible,
                    "journey_sequence": [prior.camera_id, event.camera_id]
                }

        # 2. Persist to database
        try:
            self.repo.insert_target_search_event(event)
        except Exception as p_ex:
            logger.debug(f"DB event save note: {p_ex}")

    def stop_search(self, session_id: str) -> bool:
        """Stop a running search session."""
        with self._lock:
            session = self.sessions.get(session_id)
            stop_evt = self._stop_events.get(session_id)

            if not session:
                return False

            if stop_evt:
                stop_evt.set()

            session.status = SessionStatus.STOPPED.value
            session.stopped_at = time.time()

            # Update database
            try:
                self.repo.update_target_search_session_status(session_id, SessionStatus.STOPPED.value)
            except Exception:
                pass

            return True

    def get_session(self, session_id: str) -> Optional[TargetSearchSession]:
        with self._lock:
            return self.sessions.get(session_id)

    def get_status(self, session_id: str) -> Dict[str, Any]:
        """Telemetry and metrics for a search session."""
        with self._lock:
            session = self.sessions.get(session_id)
            engine = self.engines.get(session_id)

        if not session or not engine:
            return {"error": f"Session {session_id} not found."}

        all_tracks = engine.get_all_tracks()
        confirmed_tracks = [t for t in all_tracks if t.status == CandidateStatus.CONFIRMED_CANDIDATE]
        events = engine.get_events()

        return {
            "session_id": session.session_id,
            "target_id": session.target_id,
            "name": session.name,
            "status": session.status,
            "mode": session.requested_mode,
            "active_mode": session.active_mode,
            "selected_cameras": session.selected_cameras,
            "active_cameras_count": len(session.selected_cameras),
            "candidates_count": len(all_tracks),
            "confirmed_candidates_count": len(confirmed_tracks),
            "events_count": len(events),
            "created_at": session.created_at,
            "stopped_at": session.stopped_at
        }

    def get_events(self, session_id: str) -> List[Dict[str, Any]]:
        """Return candidate events for session."""
        with self._lock:
            engine = self.engines.get(session_id)
        if not engine:
            return []
        return [e.to_dict() for e in engine.get_events()]

    def get_cameras_status(self, session_id: str) -> List[Dict[str, Any]]:
        """Return per-camera status and top candidate for a search session."""
        with self._lock:
            session = self.sessions.get(session_id)
            engine = self.engines.get(session_id)

        if not session or not engine:
            return []

        result = []
        all_tracks = engine.get_all_tracks()

        for cam_id in session.selected_cameras:
            worker = self.stream_manager.workers.get(cam_id)
            is_live = bool(worker and worker.is_running)

            cam_tracks = [t for t in all_tracks if t.camera_id == cam_id and t.status != CandidateStatus.EXPIRED]
            top_track = None
            if cam_tracks:
                # Prefer confirmed or highest confirmation score
                top_track = max(cam_tracks, key=lambda t: t.confirmation_score)

            candidate_data = None
            if top_track:
                candidate_data = {
                    "track_id": top_track.track_id,
                    "raw_similarity": round(top_track.raw_similarity_max, 4),
                    "quality_score": round(top_track.quality_mean, 4),
                    "observations_count": top_track.confirmation_count,
                    "track_consistency": round(top_track.track_consistency, 4),
                    "confirmation_score": round(top_track.confirmation_score, 4),
                    "status": top_track.status.value
                }

            result.append({
                "camera_id": cam_id,
                "is_live": is_live,
                "candidate": candidate_data
            })

        return result

    def get_evidence(self, session_id: str) -> List[Dict[str, Any]]:
        """Return evidence artifacts for session."""
        events = self.get_events(session_id)
        artifacts = []
        for e in events:
            artifacts.append({
                "event_id": e["event_id"],
                "camera_id": e["camera_id"],
                "track_id": e["track_id"],
                "confirmation_score": e["confirmation_score"],
                "best_frame_path": e["best_frame_path"],
                "person_crop_path": e["person_crop_path"],
                "face_crop_path": e["face_crop_path"],
                "hashes": e["hashes"],
                "metadata": e["metadata"]
            })
        return artifacts


# Global Singleton Coordinator
target_search_coordinator: Optional[TargetSearchCoordinator] = None


def get_target_search_coordinator(base_dir: Optional[str] = None) -> TargetSearchCoordinator:
    global target_search_coordinator
    if target_search_coordinator is None:
        target_search_coordinator = TargetSearchCoordinator(base_dir=base_dir)
    return target_search_coordinator
