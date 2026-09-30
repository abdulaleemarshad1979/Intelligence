"""
Pushkaralu Lost & Found - missing-person search for CCTV Intelligence Core.
Place at: app/missing_person/lost_found.py

FLOW
  1. Officer opens a case: photo(s) of the missing person, optional phone video,
     what they are wearing TODAY, approx height, last-seen camera/ghat + time.
  2. enroll_case() builds a QueryProfile:
       face embedding + face geometry (cheekbones, jaw, chin) + body proportions
       + gait signature + height + clothing colours.
  3. Live pipeline calls LostFoundSearch.observe(...) for every tracked person.
     Evidence is accumulated per TRACK (multi-frame), scored against ALL open
     cases, fused, and pushed to the officer review queue. Nothing auto-confirms.
  4. Case closed (person found) or TTL expires -> biometric data purged, audit kept.

MODELS (license-clean, matches repo governance table)
  Face detection : YuNet   (OpenCV Zoo, MIT)        models/face_detection_yunet_2023mar.onnx
  Face embedding : SFace   (OpenCV Zoo, Apache-2.0) models/face_recognition_sface_2021dec.onnx
  Face geometry  : MediaPipe Face Mesh / FaceLandmarker (Apache-2.0)
  Body pose      : pluggable PoseSource; default MediaPipe Pose, swap in RTMPose.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import tempfile
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Any

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

log = logging.getLogger("lost_found")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
FACE_LANDMARKER_PATH = MODEL_DIR / "face_landmarker.task"
POSE_LANDMARKER_PATH = MODEL_DIR / "pose_landmarker_lite.task"
AUDIT_LOG = BASE_DIR / "data" / "lost_found_audit.jsonl"

BBox = Tuple[float, float, float, float]  # x0, y0, x1, y1


# ============================================================ config
@dataclass
class LFConfig:
    sface_cosine_threshold: float = 0.363   # OpenCV-recommended SFace cosine threshold
    min_face_px: int = 36                   # smaller faces are skipped (unreliable)
    max_yaw_ratio: float = 0.18             # face geometry only from near-frontal faces
    ratio_veto: float = 0.30                # torso/leg disparity veto (README rule)
    height_veto_measured_cm: float = 12.0   # README rule, when height is measured
    height_veto_estimated_cm: float = 25.0  # family guesses are rough
    soft_only_cap: float = 0.45             # README rule: no face -> max 0.45
    soft_only_review_raw: float = 0.62      # raw soft score needed to reach review queue
    review_threshold: float = 0.52
    high_conf_threshold: float = 0.78
    min_track_frames: int = 8
    face_every_n: int = 3                   # run face models every N frames per track
    pose_every_n: int = 1
    case_ttl_hours: float = 72.0
    max_walk_speed_mps: float = 2.0         # camera-topology gate
    alert_cooldown_s: float = 60.0


# 14-joint layout (matches app/features/pose.py description)
J = dict(head=0, neck=1, r_sho=2, r_elb=3, r_wri=4, l_sho=5, l_elb=6, l_wri=7,
         r_hip=8, r_knee=9, r_ank=10, l_hip=11, l_knee=12, l_ank=13)


# ============================================================ face
class FaceAnalyzer:
    # MediaPipe Face Mesh landmark indices
    FOREHEAD, CHIN, NOSE = 10, 152, 1
    CHEEK_L, CHEEK_R = 234, 454      # zygomatic (cheekbone) width
    JAW_L, JAW_R = 172, 397          # jaw angle points
    EYE_L, EYE_R = 33, 263           # outer eye corners
    MOUTH_L, MOUTH_R = 61, 291

    def __init__(self, cfg: LFConfig):
        self.cfg = cfg
        if not YUNET_PATH.is_file():
            raise FileNotFoundError(f"YuNet ONNX model not found at {YUNET_PATH}")
        if not SFACE_PATH.is_file():
            raise FileNotFoundError(f"SFace ONNX model not found at {SFACE_PATH}")

        self.det = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.7, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")

        self.mesh_legacy = None
        self.mesh_tasks = None

        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self.mesh_legacy = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                    min_detection_confidence=0.5
                )
            elif hasattr(mp, "tasks") and FACE_LANDMARKER_PATH.is_file():
                from mediapipe.tasks.python import vision, BaseOptions
                options = vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_PATH)),
                    num_faces=1,
                    min_face_detection_confidence=0.5
                )
                self.mesh_tasks = vision.FaceLandmarker.create_from_options(options)
        except Exception as ex:
            log.warning("FaceAnalyzer FaceMesh backend initialization warning: %s", ex)

    def detect(self, img: np.ndarray) -> List[np.ndarray]:
        h, w = img.shape[:2]
        if h < 20 or w < 20:
            return []
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img)
        if faces is None:
            return []
        faces = [f for f in faces if min(f[2], f[3]) >= self.cfg.min_face_px]
        return sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    def embed(self, img: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        aligned = self.rec.alignCrop(img, face_row)
        f = self.rec.feature(aligned).flatten().astype(np.float32)
        return f / (np.linalg.norm(f) + 1e-9)

    def geometry(self, img: np.ndarray, face_row: np.ndarray) -> Optional[np.ndarray]:
        """Scale-invariant facial structure: cheekbones, jaw, chin, face length.
        Normalised by inter-ocular distance. Returns None for non-frontal faces."""
        x, y, w, h = face_row[:4].astype(int)
        pad = int(0.25 * max(w, h))
        H, W = img.shape[:2]
        crop = img[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
        if crop.size == 0:
            return None

        ch, cw = crop.shape[:2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        landmarks = None

        if self.mesh_legacy is not None:
            res = self.mesh_legacy.process(rgb)
            if res.multi_face_landmarks:
                landmarks = res.multi_face_landmarks[0].landmark
        elif self.mesh_tasks is not None:
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self.mesh_tasks.detect(mp_image)
            if res.face_landmarks:
                landmarks = res.face_landmarks[0]

        if landmarks is not None:
            def P(i):
                lm_i = landmarks[i]
                return np.array([lm_i.x * cw, lm_i.y * ch])

            def d(a, b):
                return float(np.linalg.norm(P(a) - P(b)))

            iod = d(self.EYE_L, self.EYE_R)
            cheek = d(self.CHEEK_L, self.CHEEK_R)
            if iod < 1 or cheek < 1:
                return None
            cheek_mid = (P(self.CHEEK_L) + P(self.CHEEK_R)) / 2
            if abs(P(self.NOSE)[0] - cheek_mid[0]) / cheek > self.cfg.max_yaw_ratio:
                return None  # head turned: ratios would be distorted
            v1, v2 = P(self.JAW_L) - P(self.CHIN), P(self.JAW_R) - P(self.CHIN)
            cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            chin_angle = math.degrees(math.acos(float(np.clip(cosang, -1, 1))))
            jaw = d(self.JAW_L, self.JAW_R)
            return np.array([
                cheek / iod,                        # cheekbone width
                jaw / iod,                          # jaw width
                jaw / cheek,                        # jaw taper (V vs square face)
                d(self.FOREHEAD, self.CHIN) / iod,  # face length
                d(self.NOSE, self.CHIN) / iod,      # lower-face / chin length
                chin_angle / 180.0,                 # chin sharpness
            ], dtype=np.float32)

        # Fallback to YuNet 5-point facial geometry if mesh unavailable
        if len(face_row) >= 14:
            # YuNet 5 keypoints: re(4,5), le(6,7), nt(8,9), rc(10,11), lc(12,13)
            re = face_row[4:6].astype(np.float32)
            le = face_row[6:8].astype(np.float32)
            nt = face_row[8:10].astype(np.float32)
            rc = face_row[10:12].astype(np.float32)
            lc = face_row[12:14].astype(np.float32)

            iod = float(np.linalg.norm(re - le))
            if iod < 2.0:
                return None
            eye_mid = (re + le) / 2.0
            mouth_mid = (rc + lc) / 2.0
            nose_to_eyes = float(np.linalg.norm(nt - eye_mid))
            mouth_width = float(np.linalg.norm(rc - lc))
            face_span = float(max(w, h))

            # Check frontal alignment
            eye_dx = abs(re[0] - le[0]) + 1e-5
            asym = abs((nt[0] - eye_mid[0]) / eye_dx)
            if asym > self.cfg.max_yaw_ratio * 1.5:
                return None

            return np.array([
                (w * 0.9) / iod,                    # estimated cheek / iod
                mouth_width * 1.4 / iod,            # estimated jaw / iod
                (mouth_width * 1.4) / (w * 0.9 + 1e-6),
                face_span / iod,
                float(np.linalg.norm(mouth_mid - nt)) / iod,
                0.55                                # canonical nominal chin taper
            ], dtype=np.float32)

        return None


def geom_similarity(a: np.ndarray, b: np.ndarray) -> float:
    rel = np.abs(a - b) / (np.abs(a) + 1e-6)
    return float(math.exp(-(float(rel.mean()) / 0.06) ** 2))  # calibrate 0.06 on field data


# ============================================================ pose / body / gait
class PoseSource(Protocol):
    def keypoints(self, img: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        """Return (14, 3) array: x, y (full-image px), confidence."""


class MediaPipePose:
    """Default MediaPipe pose source, supporting both legacy solutions and tasks vision API."""
    MP = [0, None, 12, 14, 16, 11, 13, 15, 24, 26, 28, 23, 25, 27]

    def __init__(self):
        self.pose_legacy = None
        self.pose_tasks = None
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
                self.pose_legacy = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
            elif hasattr(mp, "tasks") and POSE_LANDMARKER_PATH.is_file():
                from mediapipe.tasks.python import vision, BaseOptions
                options = vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(POSE_LANDMARKER_PATH)),
                    num_poses=1,
                    min_pose_detection_confidence=0.5
                )
                self.pose_tasks = vision.PoseLandmarker.create_from_options(options)
        except Exception as ex:
            log.warning("MediaPipePose backend initialization warning: %s", ex)

    def keypoints(self, img: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        H, W = img.shape[:2]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return None

        h, w = crop.shape[:2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        lm_list = None

        if self.pose_legacy is not None:
            r = self.pose_legacy.process(rgb)
            if r.pose_landmarks:
                lm_list = r.pose_landmarks.landmark
        elif self.pose_tasks is not None:
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self.pose_tasks.detect(mp_image)
            if res.pose_landmarks:
                lm_list = res.pose_landmarks[0]

        if not lm_list:
            return None

        out = np.zeros((14, 3), np.float32)
        for j, m in enumerate(self.MP):
            if m is not None and m < len(lm_list):
                lm = lm_list[m]
                vis = getattr(lm, "visibility", 1.0)
                if vis is None:
                    vis = 1.0
                out[j] = (lm.x * w + x0, lm.y * h + y0, vis)
        out[1] = (out[J["r_sho"]] + out[J["l_sho"]]) / 2
        return out


class RTMPoseSource:
    """Production RTMPose / YOLO-Pose source wrapping app.features.pose.PoseEstimator."""
    def __init__(self, model_name: str = "rtmpose-m", conf_threshold: float = 0.30):
        from app.features.pose import PoseEstimator
        self.estimator = PoseEstimator(model_name=model_name, conf_threshold=conf_threshold)

    def keypoints(self, img: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        H, W = img.shape[:2]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return None

        res = self.estimator.estimate_pose(crop, [x0, y0, x1, y1])
        kpts_map = res.get("keypoints_global", {})
        if not kpts_map:
            return None

        out = np.zeros((14, 3), np.float32)
        # 14 joints map from JOINT_NAMES
        joint_keys = [
            "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
            "left_shoulder", "left_elbow", "left_wrist", "right_hip",
            "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle"
        ]
        for idx, jk in enumerate(joint_keys):
            if jk in kpts_map:
                pt = kpts_map[jk]
                out[idx] = (pt[0], pt[1], pt[2] if len(pt) > 2 else 0.8)

        if out[1, 2] < 0.2:
            out[1] = (out[J["r_sho"]] + out[J["l_sho"]]) / 2
        return out


def body_ratio(kp: np.ndarray, min_conf: float = 0.4) -> Optional[float]:
    """Torso-to-leg ratio: fairly view-stable, survives clothing changes."""
    idx = [J["r_sho"], J["l_sho"], J["r_hip"], J["l_hip"], J["r_ank"], J["l_ank"]]
    if (kp[idx, 2] < min_conf).any():
        return None
    sho = (kp[J["r_sho"], :2] + kp[J["l_sho"], :2]) / 2
    hip = (kp[J["r_hip"], :2] + kp[J["l_hip"], :2]) / 2
    ank = (kp[J["r_ank"], :2] + kp[J["l_ank"], :2]) / 2
    torso, leg = np.linalg.norm(sho - hip), np.linalg.norm(hip - ank)
    return float(torso / leg) if leg > 5 and torso > 5 else None


def _interp_nan(x: np.ndarray) -> Optional[np.ndarray]:
    x = x.astype(np.float64).copy()
    bad = ~np.isfinite(x)
    if bad.all():
        return None
    i = np.arange(len(x))
    x[bad] = np.interp(i[bad], i[~bad], x[~bad])
    return x


def _detrend(x: np.ndarray, win: int) -> np.ndarray:
    win = max(3, win)
    return x - np.convolve(x, np.ones(win) / win, mode="same")


def gait_features(seq: Sequence[np.ndarray], fps: float, min_conf: float = 0.35) -> Optional[np.ndarray]:
    """Gait signature from a keypoint sequence of ONE walking person.
    [cadence (steps/s), stride amplitude, pelvic bounce, trunk lean, rhythm periodicity]"""
    if fps <= 0 or len(seq) < int(fps * 2):
        return None
    A = np.stack(seq).astype(np.float64)
    need = [J["r_ank"], J["l_ank"], J["r_hip"], J["l_hip"], J["neck"]]
    good = (A[:, need, 2] >= min_conf).all(1)
    if good.mean() < 0.6:
        return None
    A[~good, :, :2] = np.nan
    hip = (A[:, J["r_hip"], :2] + A[:, J["l_hip"], :2]) / 2
    ank = (A[:, J["r_ank"], :2] + A[:, J["l_ank"], :2]) / 2
    leg = np.nanmedian(np.linalg.norm(hip - ank, axis=1))
    if not np.isfinite(leg) or leg < 10:
        return None
    stride = _interp_nan(np.linalg.norm(A[:, J["r_ank"], :2] - A[:, J["l_ank"], :2], axis=1) / leg)
    bounce = _interp_nan(hip[:, 1] / leg)
    if stride is None or bounce is None:
        return None
    trunk = A[:, J["neck"], :2] - hip
    lean = np.nanmedian(np.abs(np.degrees(np.arctan2(trunk[:, 0], -trunk[:, 1]))))

    s = _detrend(stride, int(fps))
    spec = np.abs(np.fft.rfft(s * np.hanning(len(s))))
    freqs = np.fft.rfftfreq(len(s), 1.0 / fps)
    band = (freqs >= 0.7) & (freqs <= 3.2)
    if not band.any():
        return None
    k = int(np.argmax(np.where(band, spec, 0)))
    cadence = freqs[k]                       # inter-ankle distance peaks once per step
    periodicity = spec[k] / (spec[band].sum() + 1e-9)
    stride_amp = np.percentile(stride, 95) - np.percentile(stride, 5)
    b = _detrend(bounce, int(fps))
    bounce_amp = np.percentile(b, 95) - np.percentile(b, 5)
    return np.array([cadence, stride_amp, bounce_amp, lean / 30.0, periodicity], np.float32)


GAIT_SCALES = np.array([0.25, 0.25, 0.03, 0.15, 0.15], np.float32)  # calibrate on field data


def gait_similarity(a: np.ndarray, b: np.ndarray) -> float:
    z = (a - b) / GAIT_SCALES
    return float(math.exp(-0.5 * float(np.mean(z ** 2))))


# ============================================================ clothing
HSV_RANGES = {  # OpenCV HSV, H in 0..179
    "red": [((0, 80, 60), (8, 255, 255)), ((170, 80, 60), (179, 255, 255))],
    "saffron": [((9, 90, 80), (20, 255, 255))],
    "orange": [((9, 90, 80), (20, 255, 255))],
    "yellow": [((21, 80, 80), (34, 255, 255))],
    "green": [((35, 60, 50), (85, 255, 255))],
    "blue": [((86, 60, 50), (128, 255, 255))],
    "purple": [((129, 50, 50), (150, 255, 255))],
    "pink": [((151, 40, 80), (169, 255, 255))],
    "white": [((0, 0, 180), (179, 40, 255))],
    "black": [((0, 0, 0), (179, 255, 55))],
    "grey": [((0, 0, 56), (179, 40, 179))],
    "brown": [((5, 80, 30), (20, 255, 140))],
}


def colour_fraction(region: np.ndarray, colour: str) -> float:
    if region is None or region.size == 0 or colour not in HSV_RANGES:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in HSV_RANGES[colour]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return float(mask.mean() / 255.0)


def clothing_regions(img: np.ndarray, kp: Optional[np.ndarray], bbox: BBox):
    """Upper = shoulders->hips, lower = hips->knees. Falls back to bbox bands."""
    x0, y0, x1, y1 = [int(v) for v in bbox]
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if kp is not None and (kp[[J["r_sho"], J["l_sho"], J["r_hip"], J["l_hip"]], 2] > 0.4).all():
        ys = int(min(kp[J["r_sho"], 1], kp[J["l_sho"], 1]))
        yh = int(max(kp[J["r_hip"], 1], kp[J["l_hip"], 1]))
        yk = int(max(kp[J["r_knee"], 1], kp[J["l_knee"], 1])) if (kp[[J["r_knee"], J["l_knee"]], 2] > 0.4).all() \
            else yh + (yh - ys)
    else:
        h = y1 - y0
        ys, yh, yk = y0 + int(0.2 * h), y0 + int(0.5 * h), y0 + int(0.8 * h)
    # central 60% width, to reduce background / neighbours in dense crowds
    cx0, cx1 = x0 + int(0.2 * (x1 - x0)), x1 - int(0.2 * (x1 - x0))
    return img[max(0, ys):max(0, yh), cx0:cx1], img[max(0, yh):max(0, yk), cx0:cx1]


# ============================================================ case / profile
@dataclass
class CaseMeta:
    officer_id: str
    guardian_contact: str
    display_name: str = ""
    age: Optional[int] = None
    height_cm: Optional[float] = None
    height_is_estimate: bool = True
    upper_colours: List[str] = field(default_factory=list)   # what they wear TODAY
    lower_colours: List[str] = field(default_factory=list)
    video_is_from_today: bool = False     # only then is video clothing usable
    last_seen_cam: Optional[str] = None
    last_seen_ts: Optional[float] = None


@dataclass
class QueryProfile:
    case_id: str
    meta: CaseMeta
    face_embs: List[np.ndarray] = field(default_factory=list)
    face_geom: Optional[np.ndarray] = None
    torso_leg: Optional[float] = None
    gait: Optional[np.ndarray] = None
    created_ts: float = field(default_factory=time.time)
    expires_ts: float = 0.0
    status: str = "OPEN"  # OPEN | FOUND | EXPIRED | CANCELLED

    def modalities(self) -> Dict[str, bool]:
        m = self.meta
        return dict(face=bool(self.face_embs), face_geom=self.face_geom is not None,
                    body=self.torso_leg is not None, gait=self.gait is not None,
                    height=m.height_cm is not None,
                    clothing=bool(m.upper_colours or m.lower_colours))


def audit(event: str, **kw):
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG.open("a") as f:
        f.write(json.dumps({"ts": time.time(), "event": event, **kw}, default=str) + "\n")


def enroll_case(photos: List[np.ndarray], video_path: Optional[str], meta: CaseMeta,
                face: FaceAnalyzer, pose: PoseSource, cfg: LFConfig,
                max_video_frames: int = 450) -> QueryProfile:
    q = QueryProfile(case_id=f"LF-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}", meta=meta)
    q.expires_ts = q.created_ts + cfg.case_ttl_hours * 3600
    geoms: List[np.ndarray] = []

    def take_face(img):
        faces = face.detect(img)
        if faces:
            q.face_embs.append(face.embed(img, faces[0]))
            g = face.geometry(img, faces[0])
            if g is not None:
                geoms.append(g)

    for img in photos:
        take_face(img)

    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        kps, ratios, i = [], [], 0
        while i < max_video_frames:
            ok, frame = cap.read()
            if not ok:
                break
            H, W = frame.shape[:2]
            kp = pose.keypoints(frame, (0, 0, W, H))  # reference video: assume one person
            if kp is not None:
                kps.append(kp)
                r = body_ratio(kp)
                if r is not None:
                    ratios.append(r)
            if i % 10 == 0:
                take_face(frame)
            i += 1
        cap.release()
        q.gait = gait_features(kps, fps)
        if ratios:
            q.torso_leg = float(np.median(ratios))

    if len(q.face_embs) > 12:  # keep a diverse subset
        q.face_embs = q.face_embs[:: max(1, len(q.face_embs) // 12)][:12]
    if geoms:
        q.face_geom = np.median(np.stack(geoms), axis=0)

    audit("CASE_OPENED", case_id=q.case_id, officer=meta.officer_id,
          modalities=q.modalities(), last_seen=meta.last_seen_cam)
    if not q.face_embs:
        log.warning("%s: no usable face in photos/video; search relies on soft cues only", q.case_id)
    return q


# ============================================================ live search
@dataclass
class TrackEvidence:
    cam_id: str
    track_id: int
    first_ts: float
    last_ts: float = 0.0
    frames: int = 0
    face_embs: deque = field(default_factory=lambda: deque(maxlen=20))
    face_geoms: deque = field(default_factory=lambda: deque(maxlen=10))
    ratios: deque = field(default_factory=lambda: deque(maxlen=60))
    kps: deque = field(default_factory=lambda: deque(maxlen=150))
    heights: deque = field(default_factory=lambda: deque(maxlen=60))
    upper_crops: deque = field(default_factory=lambda: deque(maxlen=5))
    lower_crops: deque = field(default_factory=lambda: deque(maxlen=5))
    best_crop: Optional[np.ndarray] = None
    best_face_size: float = 0.0
    gait_cache: Optional[np.ndarray] = None
    gait_cache_len: int = 0


@dataclass
class Candidate:
    case_id: str
    cam_id: str
    track_id: int
    ts: float
    score: float
    raw_score: float
    tier: str
    modality_scores: Dict[str, float]
    crop_jpeg: Optional[bytes] = None


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


WEIGHTS_FACE = dict(face=0.40, face_geom=0.05, body=0.15, gait=0.20, height=0.10, clothing=0.10)
WEIGHTS_NO_FACE = dict(body=0.30, gait=0.35, height=0.15, clothing=0.20)


class LostFoundSearch:
    def __init__(self, cfg: LFConfig, face: FaceAnalyzer, pose: PoseSource,
                 height_fn: Optional[Callable[[str, BBox, Optional[np.ndarray]], Optional[float]]] = None,
                 cam_positions_m: Optional[Dict[str, Tuple[float, float]]] = None):
        """height_fn: wire to app/features/height.py (calibrated stature in cm).
        cam_positions_m: camera x/y in metres from config/cameras.yaml."""
        self.cfg, self.face, self.pose = cfg, face, pose
        self.height_fn = height_fn
        self.cam_pos = cam_positions_m or {}
        self.cases: Dict[str, QueryProfile] = {}
        self.tracks: Dict[Tuple[str, int], TrackEvidence] = {}
        self.alerts: deque = deque(maxlen=500)
        self._last_alert: Dict[Tuple[str, str, int], Tuple[float, str]] = {}

    # ---------- case lifecycle
    def open_case(self, q: QueryProfile):
        self.cases[q.case_id] = q

    def close_case(self, case_id: str, status: str, officer_id: str):
        q = self.cases.pop(case_id, None)
        if q:
            audit("CASE_CLOSED", case_id=case_id, status=status, officer=officer_id)
            q.face_embs.clear()
            q.face_geom = q.gait = None  # purge biometrics

    def expire_cases(self, now: float):
        for cid, q in list(self.cases.items()):
            if now > q.expires_ts:
                self.close_case(cid, "EXPIRED", "system")

    # ---------- per-frame ingestion (call from tracker loop)
    def observe(self, cam_id: str, track_id: int, frame: np.ndarray, bbox: BBox, ts: float, fps: float):
        if not self.cases:
            return
        key = (cam_id, track_id)
        ev = self.tracks.get(key) or TrackEvidence(cam_id, track_id, first_ts=ts)
        self.tracks[key] = ev
        ev.frames += 1
        ev.last_ts = ts
        x0, y0, x1, y1 = [int(v) for v in bbox]
        H, W = frame.shape[:2]
        crop = frame[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
        if crop.size == 0:
            return

        kp = self.pose.keypoints(frame, bbox) if ev.frames % self.cfg.pose_every_n == 0 else None
        if kp is not None:
            ev.kps.append(kp)
            r = body_ratio(kp)
            if r is not None:
                ev.ratios.append(r)
        if self.height_fn:
            h = self.height_fn(cam_id, bbox, kp)
            if h:
                ev.heights.append(h)

        if ev.frames % self.cfg.face_every_n == 0:
            head = crop[: max(1, crop.shape[0] // 3)]  # faces live in the top third
            faces = self.face.detect(head)
            if faces:
                f = faces[0]
                ev.face_embs.append(self.face.embed(head, f))
                g = self.face.geometry(head, f)
                if g is not None:
                    ev.face_geoms.append(g)
                if f[2] * f[3] > ev.best_face_size:
                    ev.best_face_size, ev.best_crop = f[2] * f[3], crop.copy()
            if ev.frames % 15 == 0:
                up, lo = clothing_regions(frame, kp, bbox)
                ev.upper_crops.append(up.copy())
                ev.lower_crops.append(lo.copy())
        if ev.best_crop is None:
            ev.best_crop = crop.copy()

        if ev.frames >= self.cfg.min_track_frames and ev.frames % 5 == 0:
            for q in self.cases.values():
                c = self._score(ev, q, fps)
                if c:
                    self._maybe_alert(c)

    def end_track(self, cam_id: str, track_id: int):
        self.tracks.pop((cam_id, track_id), None)

    # ---------- scoring
    def _reachable(self, q: QueryProfile, cam_id: str, ts: float) -> bool:
        m = q.meta
        if not (m.last_seen_cam and m.last_seen_ts and m.last_seen_cam in self.cam_pos and cam_id in self.cam_pos):
            return True
        (ax, ay), (bx, by) = self.cam_pos[m.last_seen_cam], self.cam_pos[cam_id]
        dt = ts - m.last_seen_ts
        return dt < 0 or math.hypot(ax - bx, ay - by) <= self.cfg.max_walk_speed_mps * dt + 50

    def _score(self, ev: TrackEvidence, q: QueryProfile, fps: float) -> Optional[Candidate]:
        cfg, m = self.cfg, q.meta
        if not self._reachable(q, ev.cam_id, ev.last_ts):
            return None
        s: Dict[str, float] = {}

        # hard vetoes first (README: geometric pruning gates)
        if q.torso_leg is not None and len(ev.ratios) >= 5:
            diff = abs(float(np.median(ev.ratios)) - q.torso_leg)
            if diff > cfg.ratio_veto:
                return None
            s["body"] = math.exp(-(diff / 0.08) ** 2)
        if m.height_cm is not None and len(ev.heights) >= 5:
            dh = abs(float(np.median(ev.heights)) - m.height_cm)
            veto = cfg.height_veto_estimated_cm if m.height_is_estimate else cfg.height_veto_measured_cm
            if dh > veto:
                return None
            s["height"] = math.exp(-(dh / (veto / 2)) ** 2)

        if q.face_embs and ev.face_embs:
            S = np.stack(ev.face_embs) @ np.stack(q.face_embs).T
            best = float(np.sort(S.max(1))[-3:].mean())   # top-3 frames, not a single lucky frame
            s["face"] = _sig((best - cfg.sface_cosine_threshold) / 0.05)
        if q.face_geom is not None and ev.face_geoms:
            s["face_geom"] = geom_similarity(np.median(np.stack(ev.face_geoms), 0), q.face_geom)

        if q.gait is not None and len(ev.kps) >= int(fps * 2):
            if len(ev.kps) - ev.gait_cache_len >= int(fps):
                ev.gait_cache, ev.gait_cache_len = gait_features(list(ev.kps), fps), len(ev.kps)
            if ev.gait_cache is not None:
                s["gait"] = gait_similarity(ev.gait_cache, q.gait)

        if (m.upper_colours or m.lower_colours) and ev.upper_crops:
            parts = []
            for colours, crops in ((m.upper_colours, ev.upper_crops), (m.lower_colours, ev.lower_crops)):
                if colours and crops:
                    frac = np.median([max(colour_fraction(c, col) for col in colours) for c in crops])
                    parts.append(min(1.0, frac / 0.35))  # 35%+ of region in that colour = full
            if parts:
                s["clothing"] = float(np.mean(parts))

        face_seen = "face" in s
        weights = WEIGHTS_FACE if face_seen else WEIGHTS_NO_FACE
        avail = {k: w for k, w in weights.items() if k in s}
        if not avail or sum(avail.values()) < 0.3:
            return None
        raw = sum(s[k] * w for k, w in avail.items()) / sum(avail.values())

        if face_seen:
            score = raw
            tier = ("HIGH_CONFIDENCE" if raw >= cfg.high_conf_threshold and s["face"] >= 0.7
                    else "REVIEW" if raw >= cfg.review_threshold else None)
        else:
            score = min(raw, cfg.soft_only_cap)   # soft cues never verify identity alone
            tier = "REVIEW_NO_FACE" if raw >= cfg.soft_only_review_raw else None
        if tier is None:
            return None
        return Candidate(q.case_id, ev.cam_id, ev.track_id, ev.last_ts, round(score, 3),
                         round(raw, 3), tier, {k: round(v, 3) for k, v in s.items()})

    def _maybe_alert(self, c: Candidate):
        key = (c.case_id, c.cam_id, c.track_id)
        prev = self._last_alert.get(key)
        rank = {"REVIEW_NO_FACE": 0, "REVIEW": 1, "HIGH_CONFIDENCE": 2}
        if prev and c.ts - prev[0] < self.cfg.alert_cooldown_s and rank[c.tier] <= rank[prev[1]]:
            return
        ev = self.tracks.get((c.cam_id, c.track_id))
        if ev is not None and ev.best_crop is not None:
            ok, buf = cv2.imencode(".jpg", ev.best_crop)
            c.crop_jpeg = buf.tobytes() if ok else None
        self._last_alert[key] = (c.ts, c.tier)
        self.alerts.append(c)
        audit("CANDIDATE", case_id=c.case_id, cam=c.cam_id, track=c.track_id,
              tier=c.tier, score=c.score, modalities=c.modality_scores)

    def pop_alerts(self) -> List[Candidate]:
        out = list(self.alerts)
        self.alerts.clear()
        return out


# ============================================================ helpers
def load_cam_xy(yaml_path: str | Path) -> Dict[str, Tuple[float, float]]:
    """Loads camera spatial locations from config/cameras.yaml, converting lat/lon to meters."""
    import yaml
    p = Path(yaml_path)
    if not p.is_file():
        return {}
    try:
        with open(p, "r") as f:
            data = yaml.safe_load(f) or {}
        cameras = data.get("cameras", {})
        pos_dict: Dict[str, Tuple[float, float]] = {}

        # If lat/lon present, project to local tangent plane (equirectangular)
        valid_cams = [(cid, c) for cid, c in cameras.items() if "latitude" in c and "longitude" in c]
        if valid_cams:
            ref_lat = valid_cams[0][1]["latitude"]
            ref_lon = valid_cams[0][1]["longitude"]
            lat_scale = 111320.0  # meters per degree latitude
            lon_scale = 111320.0 * math.cos(math.radians(ref_lat))
            for cid, c in valid_cams:
                dx = (float(c["longitude"]) - ref_lon) * lon_scale
                dy = (float(c["latitude"]) - ref_lat) * lat_scale
                pos_dict[cid] = (dx, dy)
        return pos_dict
    except Exception as ex:
        log.warning("Failed to load camera coordinates from %s: %s", yaml_path, ex)
        return {}


def build_calibrated_height_fn() -> Optional[Callable[[str, BBox, Optional[np.ndarray]], Optional[float]]]:
    """Wraps app/features/height.py HeightEstimator for calibrated stature."""
    try:
        from app.features.height import HeightEstimator
        estimator = HeightEstimator()

        def height_fn(cam_id: str, bbox: BBox, kp: Optional[np.ndarray]) -> Optional[float]:
            x0, y0, x1, y1 = [int(v) for v in bbox]
            foot_px = None
            head_px = None
            if kp is not None and len(kp) >= 14:
                if kp[J["head"], 2] > 0.3:
                    head_px = (float(kp[J["head"], 0]), float(kp[J["head"], 1]))
                if kp[J["r_ank"], 2] > 0.3 and kp[J["l_ank"], 2] > 0.3:
                    foot_px = (
                        float((kp[J["r_ank"], 0] + kp[J["l_ank"], 0]) / 2.0),
                        float((kp[J["r_ank"], 1] + kp[J["l_ank"], 1]) / 2.0)
                    )
            res = estimator.estimate_height_cm([x0, y0, x1, y1], foot_px=foot_px, head_px=head_px)
            return res.get("estimated_height_cm")

        return height_fn
    except Exception as ex:
        log.warning("Could not initialize calibrated height estimator: %s", ex)
        return None


# ============================================================ FastAPI router
def build_router(engine: LostFoundSearch):
    from fastapi import APIRouter, File, Form, HTTPException, UploadFile

    router = APIRouter(prefix="/api/missing", tags=["lost-found"])

    def _csv(x: str) -> List[str]:
        return [c.strip().lower() for c in x.split(",") if c.strip()]

    @router.post("/cases")
    async def open_case(
        photos: List[UploadFile] = File(...),
        video: Optional[UploadFile] = File(None),
        officer_id: str = Form(...),
        guardian_contact: str = Form(...),
        display_name: str = Form(""),
        age: Optional[int] = Form(None),
        height_cm: Optional[float] = Form(None),
        height_is_estimate: bool = Form(True),
        upper_colours: str = Form(""),
        lower_colours: str = Form(""),
        video_is_from_today: bool = Form(False),
        last_seen_cam: Optional[str] = Form(None),
        last_seen_ts: Optional[float] = Form(None)
    ):
        imgs = []
        for p in photos:
            raw = await p.read()
            if raw:
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    imgs.append(img)
        vpath = None
        if video is not None:
            raw_vid = await video.read()
            if raw_vid:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.filename or "v.mp4").suffix)
                tmp.write(raw_vid)
                tmp.close()
                vpath = tmp.name

        if not imgs and not vpath:
            raise HTTPException(400, "Need at least one photo or a video")

        meta = CaseMeta(
            officer_id=officer_id,
            guardian_contact=guardian_contact,
            display_name=display_name,
            age=age,
            height_cm=height_cm,
            height_is_estimate=height_is_estimate,
            upper_colours=_csv(upper_colours),
            lower_colours=_csv(lower_colours),
            video_is_from_today=video_is_from_today,
            last_seen_cam=last_seen_cam,
            last_seen_ts=last_seen_ts
        )
        q = enroll_case(imgs, vpath, meta, engine.face, engine.pose, engine.cfg)
        if vpath:
            Path(vpath).unlink(missing_ok=True)  # don't keep the raw family video
        engine.open_case(q)
        return {"case_id": q.case_id, "modalities": q.modalities(), "expires_ts": q.expires_ts}

    @router.get("/alerts")
    def alerts():
        return [
            dict(
                case_id=c.case_id,
                cam_id=c.cam_id,
                track_id=c.track_id,
                ts=c.ts,
                tier=c.tier,
                score=c.score,
                raw_score=c.raw_score,
                modalities=c.modality_scores,
                crop_b64=base64.b64encode(c.crop_jpeg).decode() if c.crop_jpeg else None
            )
            for c in engine.pop_alerts()
        ]

    @router.post("/cases/{case_id}/close")
    def close(case_id: str, status: str = Form("FOUND"), officer_id: str = Form(...)):
        if case_id not in engine.cases:
            raise HTTPException(404, "Unknown or already closed case")
        engine.close_case(case_id, status, officer_id)
        return {"case_id": case_id, "status": status}

    @router.get("/cases")
    def list_cases():
        return [
            dict(
                case_id=q.case_id,
                name=q.meta.display_name,
                modalities=q.modalities(),
                created_ts=q.created_ts,
                expires_ts=q.expires_ts
            )
            for q in engine.cases.values()
        ]

    return router


# ============================================================ global instance factory
_global_search_engine: Optional[LostFoundSearch] = None


def get_lost_found_engine() -> LostFoundSearch:
    global _global_search_engine
    if _global_search_engine is None:
        cfg = LFConfig()
        face = FaceAnalyzer(cfg)
        # Try RTMPose first if available, else MediaPipePose
        try:
            pose = RTMPoseSource()
        except Exception:
            pose = MediaPipePose()

        height_fn = build_calibrated_height_fn()
        cam_pos = load_cam_xy(BASE_DIR / "config" / "cameras.yaml")
        _global_search_engine = LostFoundSearch(
            cfg=cfg,
            face=face,
            pose=pose,
            height_fn=height_fn,
            cam_positions_m=cam_pos
        )
    return _global_search_engine
