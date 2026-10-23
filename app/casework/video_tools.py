"""Walking-video tools: still frames, contact sheets, slow motion and paired playback.

Timestamps come from the stored per-frame presentation times, not from frame_number / fps,
so variable-frame-rate footage is labelled with the moment it was actually shown.
"""

import subprocess
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np

from app.casework.imaging import _caption, _fit_height, _write


def _ffmpeg(args: List[str], timeout: int = 3600) -> None:
    r = subprocess.run(["ffmpeg", "-y", "-v", "error"] + args, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise ValueError(f"ffmpeg failed: {r.stderr.strip()[:300]}")


def read_frames(path: str, indices: Sequence[int]) -> Dict[int, np.ndarray]:
    """Decode the requested frame indices (decode order == presentation order for these clips)."""
    wanted, out = sorted(set(indices)), {}
    cap = cv2.VideoCapture(path)
    try:
        n = 0
        while wanted and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if n == wanted[0]:
                out[n] = frame
                wanted.pop(0)
            n += 1
    finally:
        cap.release()
    if wanted:
        raise ValueError(f"Video has no frame(s) {wanted}")
    return out


def extract_frame(path: str, dst: str, frame_no: int, pts: List[float]) -> Dict[str, Any]:
    if not 0 <= frame_no < len(pts):
        raise ValueError(f"frame_no must be 0..{len(pts) - 1}")
    _write(read_frames(path, [frame_no])[frame_no], dst)
    return {"frame_no": frame_no, "time_s": pts[frame_no]}


def contact_sheet(path: str, dst: str, pts: List[float], start_s: float, end_s: float,
                  cols: int = 4, count: int = 12, thumb_h: int = 240) -> Dict[str, Any]:
    """Evenly spaced frames from [start_s, end_s], each stamped with its frame number and time."""
    import bisect
    if end_s <= start_s or count < 1:
        raise ValueError("Need end_s > start_s and count >= 1")
    times = np.linspace(start_s, end_s, count)
    idx = sorted({min(len(pts) - 1, max(0, bisect.bisect_right(pts, t) - 1)) for t in times})
    frames = read_frames(path, idx)
    tiles = []
    for i in idx:
        t = _fit_height(frames[i], thumb_h)
        cv2.putText(t, f"#{i}  {pts[i]:.2f}s", (6, thumb_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, f"#{i}  {pts[i]:.2f}s", (6, thumb_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30)) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 30))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    _write(np.vstack(rows), dst)
    return {"start_s": start_s, "end_s": end_s, "frames": idx, "columns": cols}


def slow_motion(path: str, dst: str, start_s: float, end_s: float, speed: float = 0.25) -> Dict[str, Any]:
    """Viewing copy of an interval played at `speed` (0.1-1.0) of real time."""
    if not 0.1 <= speed <= 1.0 or end_s <= start_s:
        raise ValueError("speed must be 0.1-1.0 and end_s > start_s")
    _ffmpeg(["-ss", f"{start_s}", "-t", f"{end_s - start_s}", "-i", path, "-an",
             "-vf", f"setpts={1 / speed}*(PTS-STARTPTS)", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "20", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", dst])
    return {"start_s": start_s, "end_s": end_s, "speed": speed}


def paired_playback(path_a: str, path_b: str, dst: str, speed: float = 1.0, height: int = 480) -> Dict[str, Any]:
    """Reference and observation videos in one frame, same speed; shorter clip holds its last frame."""
    if not 0.1 <= speed <= 1.0:
        raise ValueError("speed must be 0.1-1.0")
    f = f"scale=-2:{height},setpts={1 / speed}*PTS"
    _ffmpeg(["-i", path_a, "-i", path_b, "-an", "-filter_complex",
             f"[0:v]{f}[a];[1:v]{f}[b];[a][b]hstack=inputs=2", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "22", "-pix_fmt", "yuv420p", dst])
    return {"speed": speed, "height": height}


def paired_frames(path_a: str, frame_a: int, path_b: str, frame_b: int, dst: str,
                  label_a: str, label_b: str, height: int = 640) -> Dict[str, Any]:
    a, b = read_frames(path_a, [frame_a])[frame_a], read_frames(path_b, [frame_b])[frame_b]
    a, b = _fit_height(a, height), _fit_height(b, height)
    gap = np.full((height + 28, 8, 3), 255, dtype=np.uint8)
    _write(np.hstack([_caption(a, label_a), gap, _caption(b, label_b)]), dst)
    return {"frame_a": frame_a, "frame_b": frame_b}
