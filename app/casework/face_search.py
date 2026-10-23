"""Face-similarity search that only ever produces leads for a person to review.

Nothing here decides identity or takes action. A hit is "this frame contains a face whose
similarity to the case reference is above the job's threshold", saved with its snapshot for a
reviewer to accept, reject or defer. Jobs are bound to one case, one reference image, a stated
legal authority, an explicit camera list (capped) and, for live jobs, an expiry.
"""

import bisect
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MAX_LIVE_CAMERAS = 20
MAX_LIVE_HOURS = 24.0
DEFAULT_LIVE_HOURS = 8.0
MIN_THRESHOLD = 0.363          # published SFace cosine decision point; never search below it
DEFAULT_THRESHOLD = 0.50
MAX_CANDIDATES_PER_JOB = 200
HIT_COOLDOWN_S = 30.0
SAMPLE_EVERY_S = 0.5
MODEL_FILES = ("face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx")

Face = Tuple[np.ndarray, Tuple[int, int, int, int], float]   # embedding, (x, y, w, h), detector score


class Matcher(Protocol):
    def embed_faces(self, img: np.ndarray) -> List[Face]: ...
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float: ...


def find_models_dir() -> str:
    env = os.environ.get("CCTV_MODELS_DIR")
    cands = [env] if env else []
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        cands.append(os.path.join(d, "models"))
        if os.path.dirname(d) == d:
            break
        d = os.path.dirname(d)
    for c in cands:
        if c and all(os.path.isfile(os.path.join(c, f)) for f in MODEL_FILES):
            return c
    raise FileNotFoundError(f"Face models {MODEL_FILES} not found; set CCTV_MODELS_DIR")


class SFaceMatcher:
    """OpenCV YuNet detector + SFace embedder."""

    def __init__(self, models_dir: Optional[str] = None):
        d = models_dir or find_models_dir()
        self._det = cv2.FaceDetectorYN.create(os.path.join(d, MODEL_FILES[0]), "", (320, 320), 0.6)
        self._rec = cv2.FaceRecognizerSF.create(os.path.join(d, MODEL_FILES[1]), "")
        self._lock = threading.Lock()

    def embed_faces(self, img: np.ndarray) -> List[Face]:
        with self._lock:
            self._det.setInputSize((img.shape[1], img.shape[0]))
            _, faces = self._det.detect(img)
            if faces is None:
                return []
            return [(self._rec.feature(self._rec.alignCrop(img, f)),
                     tuple(int(v) for v in f[:4]), float(f[14])) for f in faces]

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(self._rec.match(a, b, cv2.FaceRecognizerSF_FR_COSINE))


def best_hit(matcher: Matcher, ref_emb: np.ndarray, img: np.ndarray) -> Optional[Tuple[float, Tuple[int, int, int, int]]]:
    best = None
    for emb, box, _ in matcher.embed_faces(img):
        s = matcher.similarity(ref_emb, emb)
        if best is None or s > best[0]:
            best = (s, box)
    return best


def reference_embedding(matcher: Matcher, image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Reference image is unreadable")
    faces = matcher.embed_faces(img)
    if len(faces) != 1:
        raise ValueError(f"Reference must contain exactly one face (found {len(faces)}); crop it first")
    return faces[0][0]


def scan_video(matcher: Matcher, ref_emb: np.ndarray, path: str, pts: List[float], threshold: float,
               should_stop: Callable[[], bool] = lambda: False,
               on_progress: Callable[[int], None] = lambda n: None) -> List[Dict[str, Any]]:
    """Sample about every SAMPLE_EVERY_S seconds; merge hits within 2 s into the best-scoring one."""
    wanted, last_t = set(), -1e9
    for i, t in enumerate(pts):
        if t - last_t >= SAMPLE_EVERY_S - 1e-6:
            wanted.add(i)
            last_t = t
    hits: List[Dict[str, Any]] = []
    cap, n, scanned = cv2.VideoCapture(path), 0, 0
    try:
        while cap.isOpened() and not should_stop():
            ok, frame = cap.read()
            if not ok:
                break
            if n in wanted:
                scanned += 1
                h = best_hit(matcher, ref_emb, frame)
                if h and h[0] >= threshold:
                    hits.append({"frame_no": n, "time_s": pts[n] if n < len(pts) else n / 30.0, "score": h[0], "box": h[1]})
                if scanned % 20 == 0:
                    on_progress(20)
            n += 1
        on_progress(scanned % 20)
    finally:
        cap.release()
    merged: List[Dict[str, Any]] = []
    for h in hits:
        if merged and h["time_s"] - merged[-1]["time_s"] <= 2.0:
            if h["score"] > merged[-1]["score"]:
                merged[-1] = h
        else:
            merged.append(h)
    return merged


class FrameSource(Protocol):
    def grab(self, camera_id: str) -> Optional[np.ndarray]: ...
    def close(self) -> None: ...


class RtspFrameSource:
    """Keeps one RTSP capture per camera and returns its latest frame. URL comes from the camera's IP
    (CCTV_RTSP_TEMPLATE, default rtsp://{ip}:554/stream0; credentials from CCTV_RTSP_USER/PASSWORD)."""

    def __init__(self, ip_lookup: Callable[[str], Optional[str]]):
        self._ip = ip_lookup
        self._caps: Dict[str, cv2.VideoCapture] = {}

    def _url(self, camera_id: str) -> Optional[str]:
        ip = self._ip(camera_id)
        if not ip:
            return None
        url = os.environ.get("CCTV_RTSP_TEMPLATE", "rtsp://{ip}:554/stream0").format(ip=ip)
        user, pw = os.environ.get("CCTV_RTSP_USER"), os.environ.get("CCTV_RTSP_PASSWORD")
        if user and "@" not in url:
            url = url.replace("rtsp://", f"rtsp://{user}:{pw or ''}@", 1)
        return url

    def grab(self, camera_id: str) -> Optional[np.ndarray]:
        cap = self._caps.get(camera_id)
        if cap is None or not cap.isOpened():
            url = self._url(camera_id)
            if not url:
                return None
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            self._caps[camera_id] = cap
        ok, frame = cap.read()
        if not ok:
            cap.release()
            self._caps.pop(camera_id, None)
            return None
        return frame

    def close(self) -> None:
        for c in self._caps.values():
            c.release()
        self._caps.clear()


class LiveJobRunner(threading.Thread):
    """Polls the job's cameras round-robin until it expires, is stopped or hits the candidate cap."""

    def __init__(self, job: Dict[str, Any], ref_emb: np.ndarray, matcher: Matcher, source: FrameSource,
                 on_hit: Callable[[str, np.ndarray, float, Tuple[int, int, int, int]], None],
                 on_tick: Callable[[int], None], on_end: Callable[[str, Optional[str]], None],
                 cap_reached: Callable[[], bool], interval_s: float = 2.0):
        super().__init__(daemon=True, name=f"live-search-{job['job_id']}")
        self.job, self.ref, self.matcher, self.source = job, ref_emb, matcher, source
        self.on_hit, self.on_tick, self.on_end, self.cap_reached = on_hit, on_tick, on_end, cap_reached
        self.interval_s = interval_s
        self._halt = threading.Event()
        self._last_hit: Dict[str, float] = {}

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        status, err = "STOPPED", None
        try:
            cams = self.job["camera_ids"]
            while not self._halt.is_set():
                if time.time() >= self.job["expires_at"]:
                    status = "EXPIRED"
                    break
                if self.cap_reached():
                    status = "CAPPED"
                    break
                t0 = time.time()
                scanned = 0
                for cam in cams:
                    if self._halt.is_set():
                        break
                    frame = self.source.grab(cam)
                    if frame is None:
                        continue
                    scanned += 1
                    h = best_hit(self.matcher, self.ref, frame)
                    now = time.time()
                    if h and h[0] >= self.job["threshold"] and now - self._last_hit.get(cam, 0) >= HIT_COOLDOWN_S:
                        self._last_hit[cam] = now
                        self.on_hit(cam, frame, h[0], h[1])
                self.on_tick(scanned)
                self._halt.wait(max(0.0, self.interval_s - (time.time() - t0)))
        except Exception as e:  # keep the failure visible on the job instead of dying silently
            logger.exception("live search failed")
            status, err = "FAILED", str(e)
        finally:
            self.source.close()
            self.on_end(status, err)
