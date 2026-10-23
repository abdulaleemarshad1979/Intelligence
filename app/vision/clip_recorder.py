"""Short video clips around a live detection (a few seconds before and after), per camera.

Frames are kept as small JPEGs in a rolling buffer only while the face watcher is active (a target is
enrolled). A clip is started only when the watcher confirms a target match; nothing is written otherwise.
"""

import collections
import hashlib
import logging
import os
import subprocess
import threading
import time
import uuid
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class _Job:
    def __init__(self, camera_id: str, path: str, size: Tuple[int, int], fps: float, started: float, end_at: float):
        self.camera_id, self.path, self.size, self.fps = camera_id, path, size, fps
        self.started, self.end_at = started, end_at
        self.alert_ids: List[str] = []
        self.frames = 0
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
             "-framerate", f"{fps:.2f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        try:
            self.proc.stdin.write(frame.tobytes())
            self.frames += 1
        except (BrokenPipeError, ValueError):
            pass

    def finish(self) -> bool:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.wait(timeout=60)
        return self.proc.returncode == 0 and self.frames > 0 and os.path.isfile(self.path)


class ClipRecorder:
    def __init__(self, storage_dir: str, url_prefix: str = "/data/captures", pre_roll_s: float = 3.0,
                 post_roll_s: float = 5.0, max_clip_s: float = 30.0, max_width: int = 640,
                 on_finished: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.storage_dir, self.url_prefix = storage_dir, url_prefix.rstrip("/")
        self.pre, self.post, self.max_clip, self.max_width = pre_roll_s, post_roll_s, max_clip_s, max_width
        self.on_finished = on_finished
        self._buf: Dict[str, Deque[Tuple[float, bytes]]] = {}
        self._jobs: Dict[str, _Job] = {}
        self._lock = threading.Lock()
        os.makedirs(storage_dir, exist_ok=True)

    def _small(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w > self.max_width:
            frame = cv2.resize(frame, (self.max_width, int(h * self.max_width / w)))
        h, w = frame.shape[:2]
        return frame[: h - h % 2, : w - w % 2]            # yuv420p needs even dimensions

    def push(self, camera_id: str, frame: np.ndarray, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        small = self._small(frame)
        ok, jpg = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        done: List[_Job] = []
        with self._lock:
            buf = self._buf.setdefault(camera_id, collections.deque())
            buf.append((now, jpg.tobytes()))
            while buf and now - buf[0][0] > self.pre:
                buf.popleft()
            job = self._jobs.get(camera_id)
            if job:
                job.write(small)
            for cid, j in list(self._jobs.items()):
                if now >= j.end_at:
                    done.append(self._jobs.pop(cid))
        for j in done:
            self._finalize(j)

    def start(self, camera_id: str, alert_id: str, now: Optional[float] = None) -> None:
        """Begin (or extend) a clip for this camera, beginning with the buffered pre-roll."""
        now = time.time() if now is None else now
        with self._lock:
            job = self._jobs.get(camera_id)
            if job:
                job.end_at = min(now + self.post, job.started + self.max_clip)
                job.alert_ids.append(alert_id)
                return
            buf = list(self._buf.get(camera_id, []))
            if not buf:
                return
            frames = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for _, b in buf]
            span = buf[-1][0] - buf[0][0]
            fps = max(2.0, min(30.0, (len(buf) - 1) / span)) if span > 0 and len(buf) > 1 else 10.0
            size = (frames[0].shape[1], frames[0].shape[0])
            path = os.path.join(self.storage_dir, f"clip_{camera_id}_{int(now)}_{uuid.uuid4().hex[:6]}.mp4")
            job = _Job(camera_id, path, size, fps, buf[0][0], now + self.post)
            job.alert_ids.append(alert_id)
            for f in frames:
                job.write(f)
            self._jobs[camera_id] = job

    def flush(self) -> None:
        """Finish every open clip now (watchlist cleared, shutdown)."""
        with self._lock:
            jobs, self._jobs = list(self._jobs.values()), {}
            self._buf.clear()
        for j in jobs:
            self._finalize(j)

    def _finalize(self, job: _Job) -> None:
        try:
            ok = job.finish()
        except Exception as e:
            logger.warning("clip finalize failed: %s", e)
            ok = False
        if not ok:
            if os.path.exists(job.path):
                os.remove(job.path)
            return
        with open(job.path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        info = {"camera_id": job.camera_id, "alert_ids": job.alert_ids, "path": job.path,
                "url": f"{self.url_prefix}/{os.path.basename(job.path)}", "sha256": digest,
                "frames": job.frames, "fps": job.fps, "duration_s": round(job.frames / job.fps, 2)}
        if self.on_finished:
            try:
                self.on_finished(info)
            except Exception as e:
                logger.warning("clip callback failed: %s", e)
