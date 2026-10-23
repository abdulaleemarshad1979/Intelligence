"""Content-addressed, write-once media storage with probing, proxies and image quality checks."""

import hashlib
import json
import mimetypes
import os
import shutil
import statistics
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

CHUNK = 1024 * 1024
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".h264", ".264", ".ts", ".m4v", ".dav"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".tif", ".tiff"}

DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "casework", "media"
)


class MediaIntegrityError(Exception):
    """Stored bytes no longer match the recorded SHA-256."""


@dataclass
class StoredFile:
    sha256: str
    size_bytes: int
    path: str
    mime_type: str


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_kind(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in VIDEO_EXT:
        return "VIDEO"
    if ext in IMAGE_EXT:
        return "IMAGE"
    raise ValueError(f"Unsupported media type: {ext or filename!r}")


class MediaStore:
    """Stores files under <root>/<bucket>/<sha[:2]>/<sha><ext>, read-only once written."""

    def __init__(self, root: Optional[str] = None):
        self.root = os.path.abspath(root or DEFAULT_ROOT)
        os.makedirs(self.root, exist_ok=True)

    def _dest(self, bucket: str, sha: str, ext: str) -> str:
        return os.path.join(self.root, bucket, sha[:2], sha + ext.lower())

    def put(self, src_path: str, bucket: str = "originals", filename: Optional[str] = None) -> StoredFile:
        """Copy src into the store without touching the source. Identical bytes share one file."""
        ext = os.path.splitext(filename or src_path)[1]
        sha = sha256_file(src_path)
        dest = self._dest(bucket, sha, ext)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if not os.path.exists(dest):
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest), suffix=".part")
            os.close(fd)
            try:
                shutil.copyfile(src_path, tmp)
                if sha256_file(tmp) != sha:
                    raise MediaIntegrityError("Copy verification failed; source changed during ingest")
                os.chmod(tmp, 0o444)
                os.replace(tmp, dest)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        mime = mimetypes.guess_type(dest)[0] or "application/octet-stream"
        return StoredFile(sha, os.path.getsize(dest), dest, mime)

    def reserve(self, bucket: str, suffix: str) -> str:
        """Scratch path for a tool to write a derived file into; commit it with put()."""
        d = os.path.join(self.root, "_work", bucket)
        os.makedirs(d, exist_ok=True)
        fd, path = tempfile.mkstemp(dir=d, suffix=suffix)
        os.close(fd)
        return path

    def verify(self, path: str, expected_sha256: str) -> None:
        if not os.path.exists(path):
            raise MediaIntegrityError(f"Missing stored file: {path}")
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise MediaIntegrityError(f"Hash mismatch for {path}: {actual} != {expected_sha256}")


# ---------------------------------------------------------------- probing

def _run(cmd: List[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _ratio(s: Optional[str]) -> float:
    try:
        n, d = s.split("/")
        return float(n) / float(d) if float(d) else 0.0
    except Exception:
        return 0.0


def probe_video(path: str) -> Dict[str, Any]:
    """Stream facts plus the true per-frame presentation times (needed for VFR footage)."""
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
              "stream=width,height,avg_frame_rate,r_frame_rate,codec_name:format=duration",
              "-of", "json", path], 120)
    if r.returncode != 0:
        raise ValueError(f"ffprobe failed: {r.stderr.strip()[:300]}")
    info = json.loads(r.stdout)
    st = (info.get("streams") or [{}])[0]
    duration = float((info.get("format") or {}).get("duration") or 0.0)

    p = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
              "packet=pts_time", "-of", "csv=p=0", path])
    pts = sorted(float(x.split(",")[0]) for x in p.stdout.split() if x.split(",")[0] not in ("", "N/A"))
    deltas = [b - a for a, b in zip(pts, pts[1:]) if b > a]
    is_vfr = False
    if len(deltas) > 2:
        med = statistics.median(deltas)
        is_vfr = max(deltas) > med * 1.5 or min(deltas) < med * 0.5
    return {
        "width": st.get("width"), "height": st.get("height"), "codec": st.get("codec_name"),
        "duration_s": duration or (pts[-1] if pts else 0.0),
        "nominal_fps": _ratio(st.get("avg_frame_rate")) or _ratio(st.get("r_frame_rate")),
        "is_vfr": is_vfr, "frame_count": len(pts), "pts_seconds": pts,
    }


def probe_image(path: str) -> Dict[str, Any]:
    from PIL import Image
    with Image.open(path) as im:
        return {"width": im.width, "height": im.height}


def make_viewing_proxy(src: str, dst: str, max_height: int = 720) -> None:
    """H.264 viewing copy. Frame timestamps are passed through, not resampled to constant rate,
    so a position in the proxy corresponds to the same moment in the original."""
    r = _run(["ffmpeg", "-y", "-v", "error", "-i", src, "-an",
              "-vf", f"scale=-2:'min({max_height},ih)'", "-c:v", "libx264", "-preset", "veryfast",
              "-crf", "23", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
              "-movflags", "+faststart", "-f", "mp4", dst], 3600)
    if r.returncode != 0:
        raise ValueError(f"proxy generation failed: {r.stderr.strip()[:300]}")


# ---------------------------------------------------------------- quality

def assess_image_quality(path: str) -> Dict[str, Any]:
    """Advisory exposure/blur/size checks. Occlusion is not measured: a reviewer records it."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Unreadable image")
    h, w = img.shape
    blur = float(cv2.Laplacian(img, cv2.CV_64F).var())
    mean = float(img.mean())
    dark = float((img <= 5).mean())
    bright = float((img >= 250).mean())
    metrics = {"width": w, "height": h, "mean_luma": round(mean, 1),
               "sharpness_laplacian_var": round(blur, 1),
               "clipped_black_fraction": round(dark, 3), "clipped_white_fraction": round(bright, 3)}
    warnings: List[str] = []
    if mean < 60 or dark > 0.25:
        warnings.append("UNDEREXPOSED")
    if mean > 195 or bright > 0.25:
        warnings.append("OVEREXPOSED")
    if blur < 100:
        warnings.append("BLURRY")
    if min(w, h) < 200:
        warnings.append("LOW_RESOLUTION")
    warnings.append("OCCLUSION_NOT_ASSESSED")
    return {"metrics": metrics, "warnings": warnings}
