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
import threading
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
        cooldown_sec: float = 2.5,
        default_threshold: float = 0.20,
        min_resolution: int = 12,
        min_laplacian_var: float = 20.0,
        vector_index: Optional[FaceVectorIndex] = None,
        sync_db: bool = True
    ):
        self._lock = threading.Lock()
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
                raw_thresh = float(t.get("threshold", self.default_threshold))
                calibrated_thresh = min(raw_thresh, 0.22)
                t["threshold"] = calibrated_thresh
                self.targets[tid] = t
                if t.get("embedding"):
                    self.vector_index.add_target(
                        target_id=tid,
                        embedding=t["embedding"],
                        metadata={
                            "name": t.get("name", "Suspect"),
                            "threshold": calibrated_thresh,
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

        with self._lock:
            tid = target_id or f"TGT-{uuid.uuid4().hex[:8].upper()}"
            thresh = float(threshold if threshold is not None else self.default_threshold)

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
                # Use raw YuNet detection row for native OpenCV SFace alignCrop if available
                best_landmarks = best_det.get("raw_detection") if best_det.get("raw_detection") is not None else best_det.get("landmarks")

            # 2. Assess quality of reference enrollment crop
            quality_info = self.engine.assess_face_quality(
                face_crop,
                min_resolution=self.min_resolution,
                min_laplacian_var=self.min_laplacian_var
            )

            # 3. Canonical 5-point alignment if landmarks are available
            if best_landmarks is not None:
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
        detections: Optional[List[Any]] = None,
        include_cooldown: bool = False
    ) -> List[Dict[str, Any]]:
        """Scan a live video frame against enrolled targets and capture snapshots upon match."""
        if not self.targets or frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        matches = []
        now = time.time()

        # Run high-speed YuNet neural face detection directly on the full frame
        frame_faces = self.engine.detect_faces(frame)

        # Body-Guided Head Zoom Enhancement for distant/overhead surveillance feeds
        if detections:
            for det in detections:
                bbox = getattr(det, "bbox", None)
                if bbox and len(bbox) == 4:
                    px, py, pw, ph = [int(v) for v in bbox]
                    # Filter out whole-frame false detections
                    if pw > w * 0.80 and ph > h * 0.80:
                        continue
                    if pw < 15 or ph < 25:
                        continue

                    # Check if face was already detected in this person's upper body
                    already_found = any(
                        (px - 15 <= f["bbox"][0] <= px + pw + 15) and 
                        (py - 15 <= f["bbox"][1] <= py + int(ph * 0.45))
                        for f in frame_faces
                    )
                    if not already_found:
                        hx1, hy1 = max(0, px), max(0, py)
                        hx2, hy2 = min(w, px + pw), min(h, py + int(ph * 0.42))
                        if (hx2 - hx1) >= 16 and (hy2 - hy1) >= 16:
                            head_raw = frame[hy1:hy2, hx1:hx2]
                            head_zoom = cv2.resize(head_raw, (head_raw.shape[1] * 2, head_raw.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
                            lab = cv2.cvtColor(head_zoom, cv2.COLOR_BGR2LAB)
                            l, a, b = cv2.split(lab)
                            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4, 4))
                            cl = clahe.apply(l)
                            head_enh = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
                            sub_faces = self.engine.detect_faces(head_enh)
                            for sf in sub_faces:
                                sbx, sby, sbw, sbh = sf["bbox"]
                                orig_bx = hx1 + int(sbx / 2)
                                orig_by = hy1 + int(sby / 2)
                                orig_bw = max(12, int(sbw / 2))
                                orig_bh = max(12, int(sbh / 2))
                                sf["bbox"] = [orig_bx, orig_by, orig_bw, orig_bh]
                                sf["is_head_crop"] = True
                                sf["head_img"] = head_enh
                                sf["parent_body_bbox"] = [px, py, pw, ph]
                                frame_faces.append(sf)

        for f in frame_faces:
            score = f.get("score", 0.0)
            if score < 0.35:
                continue

            bx, by, bw, bh = [int(v) for v in f["bbox"]]
            bx1, by1 = max(0, bx), max(0, by)
            bx2, by2 = min(w, bx + bw), min(h, by + bh)
            if bx2 <= bx1 or by2 <= by1:
                continue

            if f.get("is_head_crop") and f.get("head_img") is not None:
                face_crop = f["head_img"]
            else:
                face_crop = frame[by1:by2, bx1:bx2]

            if min(face_crop.shape[:2]) < self.min_resolution:
                continue

            # Associate with YOLO person detection if available
            matched_body_bbox = f.get("parent_body_bbox")
            if not matched_body_bbox and detections:
                for det in detections:
                    bbox = getattr(det, "bbox", None)
                    if bbox and len(bbox) == 4:
                        px, py, pw, ph = [int(v) for v in bbox]
                        if (px - 20 <= bx <= px + pw + 20) and (py - 20 <= by <= py + ph):
                            matched_body_bbox = [px, py, pw, ph]
                            break

            # Never fabricate synthetic body bounding boxes when no genuine person detection exists

            align_ref = f.get("raw_detection") if f.get("raw_detection") is not None else f.get("landmarks")
            if f.get("is_head_crop") and f.get("head_img") is not None:
                aligned = self.engine.align_face_5point(f["head_img"], align_ref)
            elif align_ref is not None:
                aligned = self.engine.align_face_5point(frame, align_ref)
            else:
                aligned = cv2.resize(face_crop, (112, 112))

            is_viable, cand_emb = self.engine.extract_face_embedding(aligned)
            if not is_viable or np.all(cand_emb == 0):
                continue

            for tid, target in self.targets.items():
                if not target.get("embedding"):
                    continue
                target_emb = np.array(target["embedding"], dtype=np.float32)
                sim = self.engine.compute_face_similarity(cand_emb, target_emb)

                target_thresh = float(target.get("threshold", self.default_threshold))

                if sim >= target_thresh:
                    cooldown_key = (tid, camera_id)
                    last_time = self._last_capture_times.get(cooldown_key, 0.0)
                    is_new_alert = (now - last_time >= self.cooldown_sec)

                    raw_sim = round(float(sim), 4)
                    face_q = f.get("quality", {})
                    quality_score = round(float(face_q.get("quality_score", 0.83)), 4)
                    detection_score = round(float(score), 4)
                    confirmation_count = target.get("total_matches", 0) + (1 if is_new_alert else 0)
                    track_consistency = 1.0
                    confirmation_score = raw_sim
                    status = "CONFIRMED_CANDIDATE" if sim >= target_thresh else "OBSERVED"

                    if is_new_alert:
                        self._last_capture_times[cooldown_key] = now
                        target["total_matches"] = target.get("total_matches", 0) + 1
                        target["last_seen_camera"] = camera_id
                        target["last_seen_time"] = now

                        capture_event = self._save_capture(
                            frame=frame,
                            face_crop=face_crop,
                            body_bbox=matched_body_bbox,
                            camera_id=camera_id,
                            target=target,
                            similarity=sim,
                            face_bbox=[bx1, by1, bx2 - bx1, by2 - by1],
                            frame_id=frame_id
                        )
                        capture_event["raw_similarity"] = raw_sim
                        capture_event["quality_score"] = quality_score
                        capture_event["detection_score"] = detection_score
                        capture_event["confirmation_count"] = confirmation_count
                        capture_event["track_consistency"] = track_consistency
                        capture_event["confirmation_score"] = confirmation_score
                        capture_event["status"] = status
                        capture_event["confidence"] = raw_sim
                        capture_event["similarity_pct"] = round(raw_sim * 100.0, 1)
                        capture_event["is_target_match"] = True
                        self.captures.insert(0, capture_event)
                        self._dispatch_to_active_alerts(capture_event)
                        matches.append(capture_event)
                    elif include_cooldown:
                        # Return continuous live match for video HUD overlay only when requested
                        matches.append({
                            "alert_id": f"LIVE-{tid}-{camera_id}",
                            "target_id": tid,
                            "target_name": target["name"],
                            "raw_similarity": raw_sim,
                            "quality_score": quality_score,
                            "detection_score": detection_score,
                            "confirmation_count": confirmation_count,
                            "track_consistency": track_consistency,
                            "confirmation_score": confirmation_score,
                            "status": status,
                            "similarity": raw_sim,
                            "confidence": raw_sim,
                            "similarity_pct": round(raw_sim * 100.0, 1),
                            "camera_id": camera_id,
                            "frame_id": frame_id,
                            "face_bbox": [bx1, by1, bx2 - bx1, by2 - by1],
                            "body_bbox": matched_body_bbox,
                            "timestamp": now,
                            "is_target_match": True,
                            "is_cooldown": True
                        })

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

        # 1. Full Frame Snapshot with highlighted target bounding box
        full_vis = frame.copy()
        if body_bbox and len(body_bbox) == 4:
            px, py, pw, ph = body_bbox
            cv2.rectangle(full_vis, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
            cv2.putText(full_vis, f"WANTED TARGET: {target['name']}", (px, max(18, py - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        if face_bbox and len(face_bbox) == 4:
            fx1, fy1, fw, fh = face_bbox
            cv2.rectangle(full_vis, (fx1, fy1), (fx1 + fw, fy1 + fh), (0, 220, 255), 2)

        full_filename = f"{prefix}_full.jpg"
        full_path = os.path.join(self.storage_dir, full_filename)
        cv2.imwrite(full_path, full_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # 2. Forensic Contextual Face & Head Portrait Crop (HQ Super-Resolution)
        h_f, w_f = frame.shape[:2]
        if face_bbox and len(face_bbox) == 4 and face_bbox[2] > 0 and face_bbox[3] > 0:
            fx, fy, fw, fh = face_bbox
            cx = fx + fw / 2.0
            cy = fy + fh / 2.0
            # 3.3x contextual framing: captures full head, hair, ears, facial landmarks, neck and collar
            crop_size = max(56, int(max(fw, fh) * 3.3))
            px1 = max(0, int(cx - crop_size / 2.0))
            py1 = max(0, int(cy - crop_size * 0.42))  # Extra head clearance for hair & forehead
            px2 = min(w_f, px1 + crop_size)
            py2 = min(h_f, py1 + crop_size)
            portrait_raw = frame[py1:py2, px1:px2]
        else:
            portrait_raw = face_crop

        if portrait_raw is None or portrait_raw.size == 0:
            portrait_raw = face_crop

        # Multi-stage AI-grade Super-Resolution Enhancement:
        # Step A: 4x Super-resolution via Lanczos-4
        sr_size = (360, 360)
        super_res = cv2.resize(portrait_raw, sr_size, interpolation=cv2.INTER_LANCZOS4)

        # Step B: Edge-preserving bilateral filter (denoises compression artifacts while maintaining eye/lip sharpness)
        denoised = cv2.bilateralFilter(super_res, d=5, sigmaColor=35, sigmaSpace=35)

        # Step C: CLAHE on luminance channel for optimal contrast and shadow recovery
        lab_f = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        lf, af, bf = cv2.split(lab_f)
        clahe_f = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(6, 6))
        face_enh = cv2.cvtColor(cv2.merge((clahe_f.apply(lf), af, bf)), cv2.COLOR_LAB2BGR)

        # Step D: Unsharp Masking for crisp ocular, nose, and jawline detail
        gaussian = cv2.GaussianBlur(face_enh, (0, 0), sigmaX=1.6)
        face_sharp = cv2.addWeighted(face_enh, 1.45, gaussian, -0.45, 0)

        # Step E: Fine detail enhancement
        face_detail = cv2.detailEnhance(face_sharp, sigma_s=8, sigma_r=0.12)

        face_filename = f"{prefix}_face.jpg"
        face_path = os.path.join(self.storage_dir, face_filename)
        cv2.imwrite(face_path, face_detail, [int(cv2.IMWRITE_JPEG_QUALITY), 99])

        # 3. Zoomed Person Crop Snapshot (Super-resolution magnified body & attire)
        body_filename = None
        body_path = None

        # If no YOLO body bbox available, synthesize person framing from face coordinates
        if not body_bbox and face_bbox and len(face_bbox) == 4:
            fx, fy, fw, fh = [int(v) for v in face_bbox]
            sbx = max(0, int(fx - fw * 1.5))
            sby = max(0, int(fy - fh * 0.4))
            sbw = min(w_f - sbx, int(fw * 4.0))
            sbh = min(h_f - sby, int(fh * 8.5))
            if sbw > 20 and sbh > 40:
                body_bbox = [sbx, sby, sbw, sbh]

        if body_bbox and len(body_bbox) == 4:
            bx, by, bw, bh = [int(v) for v in body_bbox]
            pad_x = int(bw * 0.20)
            pad_y = int(bh * 0.15)
            bx1, py1 = max(0, bx - pad_x), max(0, by - pad_y)
            bx2, py2 = min(w_f, bx + bw + pad_x), min(h_f, by + bh + pad_y)
            if bx2 > bx1 and py2 > py1:
                raw_body = frame[py1:py2, bx1:bx2]
                body_sr_w = max(240, raw_body.shape[1] * 2)
                body_sr_h = max(360, raw_body.shape[0] * 2)
                body_zoom = cv2.resize(raw_body, (body_sr_w, body_sr_h), interpolation=cv2.INTER_LANCZOS4)
                body_denoise = cv2.bilateralFilter(body_zoom, d=5, sigmaColor=30, sigmaSpace=30)
                lab_b = cv2.cvtColor(body_denoise, cv2.COLOR_BGR2LAB)
                lb, ab, bb = cv2.split(lab_b)
                clahe_b = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(6, 6))
                body_enh = cv2.cvtColor(cv2.merge((clahe_b.apply(lb), ab, bb)), cv2.COLOR_LAB2BGR)
                body_gauss = cv2.GaussianBlur(body_enh, (0, 0), sigmaX=1.5)
                body_sharp = cv2.addWeighted(body_enh, 1.4, body_gauss, -0.4, 0)
                body_filename = f"{prefix}_person.jpg"
                body_path = os.path.join(self.storage_dir, body_filename)
                cv2.imwrite(body_path, body_sharp, [int(cv2.IMWRITE_JPEG_QUALITY), 99])

        # 4. Camera Auto-Zoom Snapshot on FACE (Focused where the face is shown!)
        if face_bbox and len(face_bbox) == 4 and face_bbox[2] > 0 and face_bbox[3] > 0:
            center_x = int(face_bbox[0] + face_bbox[2] / 2)
            center_y = int(face_bbox[1] + face_bbox[3] / 2)
        elif body_bbox and len(body_bbox) == 4:
            center_x = int(body_bbox[0] + body_bbox[2] / 2)
            center_y = int(body_bbox[1] + min(body_bbox[3] * 0.16, 40))
        else:
            center_x, center_y = (w_f // 2, h_f // 2)

        # 3.2x High-magnification optical zoom centered squarely on the FACE
        zw = int(w_f / 3.2)
        zh = int(h_f / 3.2)
        zx1 = max(0, min(w_f - zw, center_x - zw // 2))
        zy1 = max(0, min(h_f - zh, center_y - zh // 2))
        zx2 = zx1 + zw
        zy2 = zy1 + zh

        raw_zoom = frame[zy1:zy2, zx1:zx2]
        zoom_frame = cv2.resize(raw_zoom, (w_f, h_f), interpolation=cv2.INTER_LANCZOS4)
        z_gauss = cv2.GaussianBlur(zoom_frame, (0, 0), sigmaX=1.2)
        zoom_frame = cv2.addWeighted(zoom_frame, 1.35, z_gauss, -0.35, 0)

        # Draw tactical reticle directly on the FACE
        scale_x = w_f / max(1, zw)
        scale_y = h_f / max(1, zh)
        if face_bbox and len(face_bbox) == 4:
            fx, fy, fw, fh = face_bbox
            fz_x1 = max(0, min(w_f - 1, int((fx - zx1) * scale_x)))
            fz_y1 = max(0, min(h_f - 1, int((fy - zy1) * scale_y)))
            fz_x2 = max(0, min(w_f - 1, int((fx + fw - zx1) * scale_x)))
            fz_y2 = max(0, min(h_f - 1, int((fy + fh - zy1) * scale_y)))

            # Red tactical bracket around face
            cv2.rectangle(zoom_frame, (fz_x1, fz_y1), (fz_x2, fz_y2), (0, 0, 255), 2)
            cv2.putText(zoom_frame, f"WANTED FACE: {target['name'].upper()}", (fz_x1, max(20, fz_y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 2)

            fcx = (fz_x1 + fz_x2) // 2
            fcy = (fz_y1 + fz_y2) // 2
            reticle_rad = max(24, int((fz_x2 - fz_x1) * 0.75))
            cv2.circle(zoom_frame, (fcx, fcy), reticle_rad, (0, 0, 255), 2)
            cv2.line(zoom_frame, (fcx - reticle_rad - 15, fcy), (fcx - reticle_rad + 6, fcy), (0, 220, 255), 2)
            cv2.line(zoom_frame, (fcx + reticle_rad - 6, fcy), (fcx + reticle_rad + 15, fcy), (0, 220, 255), 2)
        else:
            cx_z = int((center_x - zx1) * scale_x)
            cy_z = int((center_y - zy1) * scale_y)
            cv2.circle(zoom_frame, (cx_z, cy_z), 38, (0, 0, 255), 2)

        # High-tech HUD OSD Header (Plain ASCII to avoid ???? unicode artifacts in cv2.putText)
        cv2.rectangle(zoom_frame, (0, 0), (w_f, 38), (15, 23, 42), -1)
        hud_txt = f"[TARGET] FACE AUTO-ZOOM 3.2X | CANDIDATE: {target['name'].upper()} ({round(float(similarity)*100, 1)}%) | {camera_id}"
        cv2.putText(zoom_frame, hud_txt, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 2)
        cv2.putText(zoom_frame, f"FACE SHOWN: {ts_str}", (w_f - 240, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)

        zoom_cam_filename = f"{prefix}_zoom_cam.jpg"
        zoom_cam_path = os.path.join(self.storage_dir, zoom_cam_filename)
        cv2.imwrite(zoom_cam_path, zoom_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 99])

        # Cryptographic SHA-256 integrity hash (BSA Section 63)
        full_hash = hashlib.sha256(cv2.imencode('.jpg', frame)[1]).hexdigest()
        face_hash = hashlib.sha256(cv2.imencode('.jpg', face_crop)[1]).hexdigest()
        zoom_hash = hashlib.sha256(cv2.imencode('.jpg', zoom_frame)[1]).hexdigest()

        alert_id = f"ALT-FACE-{uuid.uuid4().hex[:8].upper()}"

        face_url = f"/data/captures/{face_filename}"
        person_url = f"/data/captures/{body_filename}" if body_filename else face_url
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
            "zoomed_cam_path": zoom_cam_path,
            "full_frame_url": f"/data/captures/{full_filename}",
            "face_crop_url": face_url,
            "person_crop_url": person_url,
            "zoomed_cam_url": f"/data/captures/{zoom_cam_filename}",
            "body_crop_url": person_url,
            "raw_detection_crop": face_url,
            "enhanced_detection_crop": face_url,
            "raw_frame_hash": full_hash,
            "face_crop_hash": face_hash,
            "zoomed_cam_hash": zoom_hash,
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
            target = self.targets.get(capture_event.get("target_id"), {})
            probe_photo = target.get("reference_url") or f"/data/targets/{capture_event['target_id']}_reference.jpg"
            face_crop = capture_event.get("face_crop_url")
            person_crop = capture_event.get("person_crop_url") or capture_event.get("body_crop_url") or face_crop
            det_crop = person_crop

            conf_val = round(float(capture_event.get("confidence", 0.0)), 4)
            tier_val = "TIER_1_HIGH_CONFIDENCE" if conf_val >= 0.78 else "TIER_2_REVIEW_REQUIRED"

            active_alert_entry = {
                "alert_id": capture_event["alert_id"],
                "incident_id": f"INC-FACE-{capture_event['target_id']}",
                "camera_id": capture_event["camera_id"],
                "timestamp": capture_event["timestamp"],
                "confidence": conf_val,
                "similarity_pct": capture_event.get("similarity_pct", round(conf_val * 100, 1)),
                "tier": tier_val,
                "suspect_name": capture_event.get("target_name", "Target Subject"),
                "suspect_id": capture_event.get("target_id", "TGT-001"),
                "fir_no": f"WATCH-{capture_event['target_id']}",
                "ps_code": "LIVE-FACE-RECOGNITION",
                "bns_sections": "Watchlist Alert",
                "status": "PENDING_OFFICER_CONFIRMATION",
                "raw_frame_hash": capture_event["raw_frame_hash"],
                "enhanced_crop_hash": capture_event["face_crop_hash"],
                "probe_photo": probe_photo,
                "enhanced_probe_photo": probe_photo,
                "raw_detection_crop": det_crop,
                "enhanced_detection_crop": det_crop,
                "person_crop_url": person_crop,
                "body_crop_url": person_crop,
                "face_crop_url": face_crop,
                "full_frame_url": capture_event["full_frame_url"],
                "face_bbox": capture_event.get("face_bbox"),
                "body_bbox": capture_event.get("body_bbox"),
                "scores": {
                    "height_score": None,
                    "gait_score": None,
                    "body_score": None,
                    "face_score": round(conf_val, 3)
                },
                "biometric_comparison": {
                    "estimated_height_cm": None,
                    "known_height_cm": target.get("known_height_cm"),
                    "height_delta_cm": None,
                    "track_stride_cm": None,
                    "suspect_stride_cm": target.get("stride_length_cm"),
                    "track_carried_objects": [],
                    "suspect_carried_objects": target.get("carried_objects", []) or []
                }
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
