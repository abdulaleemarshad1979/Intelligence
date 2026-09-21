"""Real-Time Live Feed Face Watcher & Automated Snapshot Capture.

Enables enrolling target faces, continuously scanning live CCTV feeds,
and automatically capturing multi-view snapshots (scene frame + face crop)
with BSA Section 63 cryptographic hashing and instant alert dispatch.
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

logger = logging.getLogger(__name__)


class LiveFaceWatcher:
    """Watches live CCTV video streams for enrolled target faces and automatically captures matches."""

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        cooldown_sec: float = 4.0,
        default_threshold: float = 0.55
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.storage_dir = storage_dir or os.path.join(base_dir, "data", "captures")
        self.targets_dir = os.path.join(base_dir, "data", "targets")
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.targets_dir, exist_ok=True)

        self.cooldown_sec = cooldown_sec
        self.default_threshold = default_threshold
        self.engine = FaceBiometricEngine()

        # In-memory target registry: target_id -> target_data
        self.targets: Dict[str, Dict[str, Any]] = {}
        # In-memory capture log: list of match event records
        self.captures: List[Dict[str, Any]] = []
        # Debounce tracker: (target_id, camera_id) -> last_capture_time
        self._last_capture_times: Dict[Tuple[str, str], float] = {}

    def enroll_target_face(
        self,
        image_input: Union[str, np.ndarray, bytes],
        name: str,
        target_id: Optional[str] = None,
        threshold: Optional[float] = None,
        notes: str = ""
    ) -> Dict[str, Any]:
        """Enroll a target face for real-time live CCTV monitoring.

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

        # Detect face within reference image
        detected = self.engine.detect_faces(img)
        h, w = img.shape[:2]

        if detected and len(detected) > 0:
            # Pick the largest/most confident face
            best_det = max(detected, key=lambda d: d.get("score", 0.0))
            bx, by, bw, bh = [int(v) for v in best_det["bbox"]]
            bx, by = max(0, bx), max(0, by)
            bw, bh = min(bw, w - bx), min(bh, h - by)
            face_crop = img[by:by + bh, bx:bx + bw] if bw > 10 and bh > 10 else img
        else:
            bx, by, bw, bh = 0, 0, w, h
            face_crop = img

        is_viable, embedding = self.engine.extract_face_embedding(face_crop)
        if not is_viable or np.all(embedding == 0):
            # Fallback embedding on entire crop
            resized = cv2.resize(face_crop, (112, 112))
            _, embedding = self.engine.extract_face_embedding(resized)

        # Save reference face image to disk
        ref_filename = f"{tid}_reference.jpg"
        ref_path = os.path.join(self.targets_dir, ref_filename)
        cv2.imwrite(ref_path, face_crop)

        target_record = {
            "target_id": tid,
            "name": name,
            "threshold": float(thresh),
            "reference_path": ref_path,
            "reference_url": f"/data/targets/{ref_filename}",
            "embedding": embedding.tolist() if hasattr(embedding, "tolist") else list(embedding),
            "enrolled_at": time.time(),
            "notes": notes,
            "total_matches": 0,
            "last_seen_camera": None,
            "last_seen_time": None
        }

        self.targets[tid] = target_record
        logger.info(f"Enrolled target face '{name}' (ID: {tid}) with threshold {thresh:.2f}")
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
        # Either detect directly on frame or extract from person detection crops
        candidate_crops: List[Tuple[np.ndarray, List[int], Optional[List[int]]]] = []

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
                        candidate_crops.append((face_crop, [px1, py1, px2 - px1, fh], [px1, py1, px2 - px1, py2 - py1]))

        # Also run global YuNet face detection on the frame if few/no detector boxes
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
                    candidate_crops.append((fc, [bx1, by1, bx2 - bx1, by2 - by1], [b_x, by1, body_w, body_h]))

        # Match each candidate face against all enrolled targets
        now = time.time()
        for face_crop, face_bbox, body_bbox in candidate_crops:
            if face_crop.size == 0 or min(face_crop.shape[:2]) < 12:
                continue

            _, cand_emb = self.engine.extract_face_embedding(face_crop)
            if np.all(cand_emb == 0):
                continue

            for tid, target in self.targets.items():
                target_emb = np.array(target["embedding"], dtype=np.float32)
                sim = self.engine.compute_face_similarity(cand_emb, target_emb)

                if sim >= target["threshold"]:
                    # Cooldown check: avoid spamming captures on consecutive frames of the same sighting
                    cooldown_key = (tid, camera_id)
                    last_time = self._last_capture_times.get(cooldown_key, 0.0)

                    if now - last_time >= self.cooldown_sec:
                        self._last_capture_times[cooldown_key] = now
                        target["total_matches"] += 1
                        target["last_seen_camera"] = camera_id
                        target["last_seen_time"] = now

                        # AUTOMATIC CAPTURE: Save full scene + face crop + body crop
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

                        # Broadcast to Command Center Active Alerts
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

        return {
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

    def _dispatch_to_active_alerts(self, capture_event: Dict[str, Any]):
        """Inject into the platform's ACTIVE_ALERTS store for real-time UI notification."""
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
            logger.debug(f"Alert dispatch note: {ex}")

    def get_targets(self) -> List[Dict[str, Any]]:
        return list(self.targets.values())

    def remove_target(self, target_id: str) -> bool:
        if target_id in self.targets:
            del self.targets[target_id]
            return True
        return False

    def get_captures(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.captures[:limit]

    def clear_targets(self):
        self.targets.clear()
        self._last_capture_times.clear()

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
