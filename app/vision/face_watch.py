"""Real-Time Live Feed Face Watcher & Automated Snapshot Capture.

Enables enrolling target faces, continuously scanning live CCTV feeds,
and automatically capturing multi-view snapshots (scene frame + face crop)
with BSA Section 63 cryptographic hashing and instant multi-channel alert dispatch.

Pipeline Architecture:
[ CCTV RTSP Feed ] ──► [ Face Detection ] ──► [ Quality Gate & 5-Pt Align ] ──► [ Feature Extraction ] ──► [ 1:N Vector Search ] ──► [ Multi-Channel Alert & Snapshots ]
"""

import os
import time
import uuid
import base64
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
import numpy as np

from app.vision.face_engine import FaceBiometricEngine
from app.vision.vector_search import FaceVectorIndex, face_vector_index
from app.integrations.notifications import notification_dispatcher
from app.database.postgres import postgres_db

logger = logging.getLogger(__name__)


class LiveFaceWatcher:
    """Watches live CCTV video streams for enrolled target faces and automatically captures matches."""

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        cooldown_sec: float = 30.0,
        default_threshold: float = 0.55,
        min_resolution: int = 24,
        min_laplacian_var: float = 35.0,
        vector_index: Optional[FaceVectorIndex] = None,
        sync_db: bool = True
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.storage_dir = storage_dir or os.path.join(base_dir, "data", "captures")
        self.targets_dir = os.path.join(base_dir, "data", "targets")
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.targets_dir, exist_ok=True)

        self.cooldown_sec = cooldown_sec
        self.default_threshold = default_threshold
        self.min_resolution = min_resolution
        self.min_laplacian_var = min_laplacian_var
        self.sync_db = sync_db

        self.engine = FaceBiometricEngine()
        self.vector_index = vector_index if vector_index is not None else FaceVectorIndex()

        # In-memory target registry: target_id -> target_data
        self.targets: Dict[str, Dict[str, Any]] = {}
        # In-memory capture log: list of match event records
        self.captures: List[Dict[str, Any]] = []
        # Debounce tracker: (target_id, camera_id) -> last_capture_time
        self._last_capture_times: Dict[Tuple[str, str], float] = {}

        # Hydrate from PostgreSQL database if targets exist
        if self.sync_db:
            self._load_targets_from_database()

    def _load_targets_from_database(self):
        if not self.sync_db:
            return
        try:
            db_targets = postgres_db.get_all_suspects()
            for t in db_targets:
                tid = t.get("target_id")
                if not tid:
                    continue
                self.targets[tid] = t
                if t.get("embedding"):
                    self.vector_index.add_target(
                        target_id=tid,
                        embedding=t["embedding"],
                        metadata={
                            "name": t.get("name", "Suspect"),
                            "threshold": float(t.get("threshold", self.default_threshold)),
                            "notes": t.get("notes", ""),
                            "reference_url": t.get("reference_url", "")
                        }
                    )
            if db_targets:
                logger.info(f"Loaded {len(db_targets)} suspects from PostgreSQL into live watchlist.")
        except Exception as ex:
            logger.debug(f"Database hydration note: {ex}")

    @staticmethod
    def _get_image_base64(img: np.ndarray) -> str:
        if img is None or img.size == 0:
            return ""
        try:
            _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
        except Exception:
            return ""

    def enroll_target_face(
        self,
        image_input: Union[str, np.ndarray, bytes],
        name: str,
        target_id: Optional[str] = None,
        threshold: Optional[float] = None,
        notes: str = ""
    ) -> Dict[str, Any]:
        """Enroll a target face for real-time live CCTV monitoring with quality diagnostics.

        image_input can be:
        - File path to image
        - Base64-encoded image string
        - Raw image bytes
        - cv2/numpy BGR image array
        """
        img = self._decode_image(image_input)
        if img is None or img.size == 0:
            raise ValueError("Invalid or unreadable target face image.")

        tid = target_id or f"TGT-{uuid.uuid4().hex[:8].upper()}"
        thresh = threshold if threshold is not None else self.default_threshold

        h, w = img.shape[:2]

        # 1. Detect face and landmarks within reference image
        detected = self.engine.detect_faces(img)
        best_landmarks = None
        face_crop = img

        if detected and len(detected) > 0:
            # Pick the largest/most confident face
            best_det = max(detected, key=lambda d: d.get("score", 0.0))
            bx, by, bw, bh = [int(v) for v in best_det["bbox"]]
            bx, by = max(0, bx), max(0, by)
            bw, bh = min(bw, w - bx), min(bh, h - by)
            if bw > 10 and bh > 10:
                face_crop = img[by:by + bh, bx:bx + bw]
            best_landmarks = best_det.get("landmarks")

        # 2. Assess quality of reference enrollment crop
        quality_info = self.engine.assess_face_quality(
            face_crop,
            min_resolution=self.min_resolution,
            min_laplacian_var=self.min_laplacian_var
        )

        # 3. Canonical 5-point alignment if landmarks are available
        if best_landmarks:
            aligned_face = self.engine.align_face_5point(img, best_landmarks)
        else:
            aligned_face = cv2.resize(face_crop, (112, 112))

        # 4. Extract biometric feature embedding
        is_viable, embedding = self.engine.extract_face_embedding(aligned_face)
        if not is_viable or np.all(embedding == 0):
            # Fallback on raw crop
            resized = cv2.resize(face_crop, (112, 112))
            _, embedding = self.engine._extract_structural_fallback(resized)

        # 5. Save reference face image to disk
        ref_filename = f"{tid}_reference.jpg"
        ref_path = os.path.join(self.targets_dir, ref_filename)
        cv2.imwrite(ref_path, face_crop)

        emb_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

        target_record = {
            "target_id": tid,
            "name": name,
            "threshold": float(thresh),
            "reference_path": ref_path,
            "reference_url": f"/data/targets/{ref_filename}",
            "embedding": emb_list,
            "enrolled_at": time.time(),
            "notes": notes,
            "quality": quality_info,
            "total_matches": 0,
            "last_seen_camera": None,
            "last_seen_time": None
        }

        self.targets[tid] = target_record

        # 6. Index into fast 1:N vector database
        self.vector_index.add_target(
            target_id=tid,
            embedding=embedding,
            metadata={
                "name": name,
                "threshold": float(thresh),
                "notes": notes,
                "reference_url": f"/data/targets/{ref_filename}"
            }
        )

        # 7. Persist target to PostgreSQL database
        if self.sync_db:
            try:
                target_record["image_base64"] = self._get_image_base64(face_crop)
                postgres_db.save_suspect(target_record)
            except Exception as p_ex:
                logger.debug(f"PostgreSQL target save note: {p_ex}")

        logger.info(f"Enrolled target face '{name}' (ID: {tid}) with threshold {thresh:.2f}, indexed in PostgreSQL & 1:N vector DB")
        return target_record

    def process_frame(
        self,
        frame: np.ndarray,
        camera_id: str = "CAM-001",
        frame_id: int = 0,
        detections: Optional[List[Any]] = None
    ) -> List[Dict[str, Any]]:
        """Scan a live video frame against enrolled targets and capture snapshots upon match."""
        if not self.targets or frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        matches = []

        # Candidate face regions to inspect:
        # Tuple of: (face_crop, face_bbox, body_bbox, landmarks)
        candidate_crops: List[Tuple[np.ndarray, List[int], Optional[List[int]], Optional[List[Any]]]] = []

        if detections:
            # Check faces located inside detected person boxes
            for det in detections:
                bbox = getattr(det, "bbox", None)
                if bbox and len(bbox) == 4:
                    px, py, pw, ph = [int(v) for v in bbox]
                    px1, py1 = max(0, px), max(0, py)
                    px2, py2 = min(w, px + pw), min(h, py + ph)
                    if px2 > px1 and py2 > py1:
                        person_crop = frame[py1:py2, px1:px2]
                        # Upper 30% of body is face region
                        fh = max(16, int((py2 - py1) * 0.30))
                        face_crop = person_crop[0:fh, :]
                        candidate_crops.append((
                            face_crop,
                            [px1, py1, px2 - px1, fh],
                            [px1, py1, px2 - px1, py2 - py1],
                            None
                        ))

        # Also run YuNet neural face detection if few/no detector person boxes
        if len(candidate_crops) == 0:
            frame_faces = self.engine.detect_faces(frame)
            for f in frame_faces:
                bx, by, bw, bh = [int(v) for v in f["bbox"]]
                bx1, by1 = max(0, bx), max(0, by)
                bx2, by2 = min(w, bx + bw), min(h, by + bh)
                if bx2 > bx1 and by2 > by1:
                    fc = frame[by1:by2, bx1:bx2]
                    # Estimate approximate body box below head
                    body_h = min(h - by1, int(bh * 4.5))
                    body_w = min(w - bx1, int(bw * 1.8))
                    b_x = max(0, bx1 - int(bw * 0.4))
                    candidate_crops.append((
                        fc,
                        [bx1, by1, bx2 - bx1, by2 - by1],
                        [b_x, by1, body_w, body_h],
                        f.get("landmarks")
                    ))

        now = time.time()

        # Match each candidate face against 1:N vector index
        for face_crop, face_bbox, body_bbox, landmarks in candidate_crops:
            if face_crop.size == 0:
                continue

            # Quality Gating Check: filter out motion blur or low resolution crops
            if min(face_crop.shape[:2]) < self.min_resolution:
                continue

            q_info = self.engine.assess_face_quality(
                face_crop,
                min_resolution=self.min_resolution,
                min_laplacian_var=self.min_laplacian_var
            )
            if not q_info["is_viable"]:
                # Drop if motion blurred below threshold or too small
                if q_info["laplacian_var"] < self.min_laplacian_var or min(face_crop.shape[:2]) < self.min_resolution:
                    continue

            # Extract 5-point aligned crop if landmarks available
            if landmarks:
                aligned = self.engine.align_face_5point(frame, landmarks)
            else:
                aligned = cv2.resize(face_crop, (112, 112))

            # Extract candidate embedding
            is_viable, cand_emb = self.engine.extract_face_embedding(aligned)
            if not is_viable or np.all(cand_emb == 0):
                continue

            # 1:N Vector Search in sub-millisecond time
            # Find closest candidate identities from watchlist
            search_results = self.vector_index.search(
                query_vector=cand_emb,
                top_k=3,
                threshold=0.30
            )

            for hit in search_results:
                tid = hit["target_id"]
                target = self.targets.get(tid)
                if not target:
                    continue

                sim = hit["similarity"]
                # Verify individual target threshold
                if sim >= target["threshold"]:
                    # Cooldown Debounce Gate: avoid notification spamming
                    cooldown_key = (tid, camera_id)
                    last_time = self._last_capture_times.get(cooldown_key, 0.0)

                    if now - last_time >= self.cooldown_sec:
                        self._last_capture_times[cooldown_key] = now
                        target["total_matches"] += 1
                        target["last_seen_camera"] = camera_id
                        target["last_seen_time"] = now

                        # AUTOMATIC SNAPSHOT CAPTURE: Full scene + face crop + body crop + SHA-256
                        capture_event = self._save_capture(
                            frame=frame,
                            face_crop=face_crop,
                            body_bbox=body_bbox,
                            camera_id=camera_id,
                            target=target,
                            similarity=sim,
                            face_bbox=face_bbox,
                            frame_id=frame_id
                        )
                        matches.append(capture_event)
                        self.captures.insert(0, capture_event)

                        # Multi-Channel Alert Dispatch (WebSockets, Telegram, SMS, Webhook, Audit Log)
                        self._dispatch_to_active_alerts(capture_event)

        return matches

    def _save_capture(
        self,
        frame: np.ndarray,
        face_crop: np.ndarray,
        body_bbox: Optional[List[int]],
        camera_id: str,
        target: Dict[str, Any],
        similarity: float,
        face_bbox: List[int],
        frame_id: int
    ) -> Dict[str, Any]:
        """Save high-resolution full scene and crops with cryptographic hash integrity."""
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        unique_suffix = uuid.uuid4().hex[:6]
        prefix = f"{camera_id}_{target['target_id']}_{ts_str}_{unique_suffix}"

        # 1. Full Frame Snapshot
        full_filename = f"{prefix}_full.jpg"
        full_path = os.path.join(self.storage_dir, full_filename)
        cv2.imwrite(full_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # 2. Face Crop Snapshot
        face_filename = f"{prefix}_face.jpg"
        face_path = os.path.join(self.storage_dir, face_filename)
        cv2.imwrite(face_path, face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # 3. Body Crop Snapshot (if available)
        body_filename = None
        body_path = None
        if body_bbox and len(body_bbox) == 4:
            bx, by, bw, bh = body_bbox
            h, w = frame.shape[:2]
            bx1, py1 = max(0, bx), max(0, by)
            bx2, py2 = min(w, bx + bw), min(h, by + bh)
            if bx2 > bx1 and py2 > py1:
                body_crop = frame[py1:py2, bx1:bx2]
                body_filename = f"{prefix}_body.jpg"
                body_path = os.path.join(self.storage_dir, body_filename)
                cv2.imwrite(body_path, body_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Cryptographic SHA-256 integrity hash (BSA Section 63)
        full_hash = hashlib.sha256(cv2.imencode('.jpg', frame)[1]).hexdigest()
        face_hash = hashlib.sha256(cv2.imencode('.jpg', face_crop)[1]).hexdigest()

        alert_id = f"ALT-FACE-{uuid.uuid4().hex[:8].upper()}"

        capture_rec = {
            "alert_id": alert_id,
            "target_id": target["target_id"],
            "target_name": target["name"],
            "camera_id": camera_id,
            "frame_id": frame_id,
            "timestamp": time.time(),
            "confidence": round(float(similarity), 4),
            "similarity_pct": round(float(similarity) * 100.0, 1),
            "face_bbox": face_bbox,
            "body_bbox": body_bbox,
            "full_frame_path": full_path,
            "face_crop_path": face_path,
            "body_crop_path": body_path,
            "full_frame_url": f"/data/captures/{full_filename}",
            "face_crop_url": f"/data/captures/{face_filename}",
            "body_crop_url": f"/data/captures/{body_filename}" if body_filename else None,
            "raw_frame_hash": full_hash,
            "face_crop_hash": face_hash,
            "notes": target.get("notes", "")
        }

        # Persist capture record to PostgreSQL
        if self.sync_db:
            try:
                postgres_db.save_capture(capture_rec)
            except Exception as p_ex:
                logger.debug(f"PostgreSQL capture save note: {p_ex}")

        return capture_rec

    def _dispatch_to_active_alerts(self, capture_event: Dict[str, Any]):
        """Inject into ACTIVE_ALERTS store and broadcast via Multi-Channel Dispatcher."""
        try:
            from app.api.routes_alerts import ACTIVE_ALERTS
            active_alert_entry = {
                "alert_id": capture_event["alert_id"],
                "incident_id": f"INC-FACE-{capture_event['target_id']}",
                "camera_id": capture_event["camera_id"],
                "timestamp": capture_event["timestamp"],
                "confidence": capture_event["confidence"],
                "tier": "TIER_1_HIGH_CONFIDENCE" if capture_event["confidence"] >= 0.70 else "TIER_2_CANDIDATE",
                "suspect_name": capture_event["target_name"],
                "fir_no": f"WATCH-{capture_event['target_id']}",
                "ps_code": "LIVE-FACE-RECOGNITION",
                "bns_sections": "Watchlist Alert",
                "status": "PENDING_DUAL_SIGNOFF",
                "raw_frame_hash": capture_event["raw_frame_hash"],
                "enhanced_crop_hash": capture_event["face_crop_hash"],
                "raw_detection_crop": capture_event["face_crop_url"],
                "full_frame_url": capture_event["full_frame_url"]
            }
            ACTIVE_ALERTS.insert(0, active_alert_entry)
        except Exception as ex:
            logger.debug(f"Alert store insert note: {ex}")

        # Multi-Channel Dispatch: WebSockets, Telegram, SMS, Webhook
        try:
            notification_dispatcher.dispatch_alert(capture_event)
        except Exception as ex:
            logger.debug(f"Notification dispatcher note: {ex}")

    def probe_image(
        self,
        image_input: Union[str, np.ndarray, bytes],
        top_k: int = 5,
        threshold: float = 0.40
    ) -> List[Dict[str, Any]]:
        """1:N forensic probe: Compare an arbitrary image against the enrolled watchlist."""
        img = self._decode_image(image_input)
        if img is None or img.size == 0:
            return []

        detected = self.engine.detect_faces(img)
        if not detected:
            return []

        results = []
        for face in detected:
            bx, by, bw, bh = [int(v) for v in face["bbox"]]
            h, w = img.shape[:2]
            bx1, py1 = max(0, bx), max(0, by)
            bx2, py2 = min(w, bx + bw), min(h, by + bh)
            face_crop = img[py1:py2, bx1:bx2]
            is_viable, emb = self.engine.extract_face_embedding(face_crop)
            if is_viable:
                hits = self.vector_index.search(emb, top_k=top_k, threshold=threshold)
                results.append({
                    "face_bbox": face["bbox"],
                    "quality": face.get("quality", {}),
                    "matches": hits
                })
        return results

    def get_targets(self) -> List[Dict[str, Any]]:
        return list(self.targets.values())

    def remove_target(self, target_id: str) -> bool:
        if target_id in self.targets:
            del self.targets[target_id]
            self.vector_index.remove_target(target_id)
            if self.sync_db:
                try:
                    postgres_db.delete_suspect(target_id)
                except Exception as p_ex:
                    logger.debug(f"PostgreSQL target delete note: {p_ex}")
            return True
        return False

    def get_captures(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.captures[:limit]

    def clear_targets(self):
        self.targets.clear()
        self.vector_index.clear()
        self._last_capture_times.clear()
        if self.sync_db:
            try:
                postgres_db.clear_all()
            except Exception as p_ex:
                logger.debug(f"PostgreSQL target clear note: {p_ex}")

    def update_settings(
        self,
        cooldown_sec: Optional[float] = None,
        default_threshold: Optional[float] = None,
        min_resolution: Optional[int] = None,
        min_laplacian_var: Optional[float] = None
    ):
        if cooldown_sec is not None:
            self.cooldown_sec = float(cooldown_sec)
        if default_threshold is not None:
            self.default_threshold = float(default_threshold)
        if min_resolution is not None:
            self.min_resolution = int(min_resolution)
        if min_laplacian_var is not None:
            self.min_laplacian_var = float(min_laplacian_var)

    def get_status(self) -> Dict[str, Any]:
        return {
            "biometric_backend": self.engine.active_backend,
            "embedding_dimension": self.engine.embedding_dim,
            "cooldown_sec": self.cooldown_sec,
            "default_threshold": self.default_threshold,
            "min_face_resolution": self.min_resolution,
            "min_laplacian_var": self.min_laplacian_var,
            "enrolled_targets_count": len(self.targets),
            "vector_search": self.vector_index.get_stats(),
            "notifications": notification_dispatcher.get_status(),
            "database": postgres_db.get_status()
        }

    def _decode_image(self, image_input: Union[str, np.ndarray, bytes]) -> Optional[np.ndarray]:
        """Helper to convert various image formats to cv2 BGR ndarray."""
        if isinstance(image_input, np.ndarray):
            return image_input

        if isinstance(image_input, bytes):
            return cv2.imdecode(np.frombuffer(image_input, np.uint8), cv2.IMREAD_COLOR)

        if isinstance(image_input, str):
            # Check if file path
            if os.path.isfile(image_input):
                return cv2.imread(image_input)
            # Check if base64 data URI or raw base64
            clean_b64 = image_input.split(",", 1)[1] if "," in image_input else image_input
            try:
                data = base64.b64decode(clean_b64)
                return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                return None

        return None


# Global singleton instance for live stream ingestion & API access
live_face_watcher = LiveFaceWatcher()
