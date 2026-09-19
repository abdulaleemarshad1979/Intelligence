"""Track manager buffering multi-frame evidence and selecting best quality frames."""

import os
import cv2
import time
import numpy as np
from typing import Dict, List, Any, Optional

def compute_sharpness(img: np.ndarray) -> float:
    """Compute image sharpness using Laplacian variance."""
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

class TrackBuffer:
    def __init__(self, track_id: str, camera_id: str):
        self.track_id = track_id
        self.camera_id = camera_id
        self.first_seen = time.time()
        self.last_seen = self.first_seen
        self.frame_indices: List[int] = []
        self.boxes: List[List[int]] = []
        self.crops: List[np.ndarray] = []
        self.sharpness_scores: List[float] = []
        self.best_crop: Optional[np.ndarray] = None
        self.best_sharpness: float = -1.0
        self.best_frame_idx: int = 0
        self.max_history = 60

    def add_observation(self, frame_idx: int, box: List[int], full_frame: np.ndarray):
        self.last_seen = time.time()
        self.frame_indices.append(frame_idx)
        self.boxes.append(box)

        # Extract person crop
        x1, y1, x2, y2 = box
        h, w = full_frame.shape[:2]
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(w, x2), min(h, y2)

        crop = full_frame[cy1:cy2, cx1:cx2].copy()
        sharpness = compute_sharpness(crop)

        self.crops.append(crop)
        self.sharpness_scores.append(sharpness)

        if sharpness > self.best_sharpness:
            self.best_sharpness = sharpness
            self.best_crop = crop
            self.best_frame_idx = frame_idx

        # Keep buffer bounded
        if len(self.crops) > self.max_history:
            self.crops.pop(0)
            self.boxes.pop(0)
            self.frame_indices.pop(0)
            self.sharpness_scores.pop(0)

class TrackManager:
    def __init__(self, storage_dir: str = "data/tracks"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.active_tracks: Dict[str, TrackBuffer] = {}

    def get_or_create(self, track_id: str, camera_id: str) -> TrackBuffer:
        if track_id not in self.active_tracks:
            self.active_tracks[track_id] = TrackBuffer(track_id, camera_id)
        return self.active_tracks[track_id]

    def record_frame(self, track_id: str, camera_id: str, frame_idx: int, box: List[int], full_frame: np.ndarray):
        tb = self.get_or_create(track_id, camera_id)
        tb.add_observation(frame_idx, box, full_frame)

    def save_best_crop(self, track_id: str) -> Optional[str]:
        if track_id not in self.active_tracks:
            return None
        tb = self.active_tracks[track_id]
        if tb.best_crop is None or tb.best_crop.size == 0:
            return None

        track_folder = os.path.join(self.storage_dir, track_id)
        os.makedirs(track_folder, exist_ok=True)
        save_path = os.path.join(track_folder, f"best_frame_{tb.best_frame_idx}.jpg")
        cv2.imwrite(save_path, tb.best_crop)
        return save_path
