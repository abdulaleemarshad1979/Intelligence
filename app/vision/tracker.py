"""ByteTrack Intra-Camera Multi-Object Tracker with Kalman State Estimation.

Implements two-stage Hungarian association:
- Stage 1: High-confidence detections (score >= 0.6) matched to active Kalman tracklets.
- Stage 2: Low-confidence detections (0.2 <= score < 0.6) matched to remaining unmatched tracklets,
  recovering targets undergoing partial occlusion or motion blur without creating false fragments.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from app.vision.detector import Detection


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Calculates Intersection over Union (IoU) between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union_area = boxAArea + boxBArea - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


@dataclass
class Tracklet:
    track_id: int
    camera_id: str
    t_in: float
    t_out: float
    bbox: Tuple[float, float, float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)
    hits: int = 1
    time_since_update: int = 0
    state: str = "TRACKED"  # TRACKED, LOST, REMOVED
    history: List[Tuple[float, float, float, float]] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    active_attributes: List[str] = field(default_factory=list)

    def predict(self):
        """Simple constant-velocity Kalman-style prediction step."""
        vx, vy = self.velocity
        x1, y1, x2, y2 = self.bbox
        self.bbox = (x1 + vx, y1 + vy, x2 + vx, y2 + vy)
        self.time_since_update += 1

    def update(self, detection: Detection, timestamp: float):
        """Update tracklet state with matched detection."""
        old_cx = (self.bbox[0] + self.bbox[2]) / 2.0
        old_cy = (self.bbox[1] + self.bbox[3]) / 2.0

        new_cx = (detection.bbox[0] + detection.bbox[2]) / 2.0
        new_cy = (detection.bbox[1] + detection.bbox[3]) / 2.0

        # Update smooth velocity vector
        alpha = 0.6
        self.velocity = (
            alpha * (new_cx - old_cx) + (1.0 - alpha) * self.velocity[0],
            alpha * (new_cy - old_cy) + (1.0 - alpha) * self.velocity[1]
        )

        self.bbox = detection.bbox
        self.history.append(detection.bbox)
        if len(self.history) > 60:
            self.history.pop(0)

        self.t_out = timestamp
        self.hits += 1
        self.time_since_update = 0
        self.state = "TRACKED"


class ByteTrackTracker:
    """Production ByteTrack intra-camera tracker."""

    def __init__(
        self,
        camera_id: str = "CAM-001",
        high_thresh: float = 0.60,
        low_thresh: float = 0.20,
        match_thresh: float = 0.70,
        max_time_lost: int = 30
    ):
        self.camera_id = camera_id
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost

        self._next_id = 1
        self.tracked_stracks: List[Tracklet] = []
        self.lost_stracks: List[Tracklet] = []
        self.removed_stracks: List[Tracklet] = []
        self.frame_count = 0

    def update(self, detections: List[Detection], timestamp: Optional[float] = None) -> List[Tracklet]:
        """Update tracker with frame detections using ByteTrack two-stage association."""
        self.frame_count += 1
        ts = timestamp if timestamp is not None else time.time()

        # Split detections into high-confidence and low-confidence
        dets_high = [d for d in detections if d.confidence >= self.high_thresh]
        dets_low = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]

        # Step 1: Predict new locations of active tracklets
        for track in self.tracked_stracks:
            track.predict()
        for track in self.lost_stracks:
            track.predict()

        # Step 2: First association with high-score detections
        unconfirmed_tracks = [t for t in self.tracked_stracks if t.hits < 2]
        confirmed_tracks = [t for t in self.tracked_stracks if t.hits >= 2]
        pool = confirmed_tracks + self.lost_stracks

        matched_a, unmatched_tracks_a, unmatched_dets_a = self._associate(pool, dets_high, iou_thresh=0.3)

        for t_idx, d_idx in matched_a:
            pool[t_idx].update(dets_high[d_idx], ts)
            if pool[t_idx] in self.lost_stracks:
                self.lost_stracks.remove(pool[t_idx])
                self.tracked_stracks.append(pool[t_idx])

        # Step 3: Second association with low-score detections (recovering occluded tracks)
        remaining_tracks = [pool[i] for i in unmatched_tracks_a if pool[i].state == "TRACKED"]
        matched_b, unmatched_tracks_b, _ = self._associate(remaining_tracks, dets_low, iou_thresh=0.5)

        for t_idx, d_idx in matched_b:
            remaining_tracks[t_idx].update(dets_low[d_idx], ts)

        # Mark tracks that failed both associations as LOST
        for t_idx in unmatched_tracks_b:
            track = remaining_tracks[t_idx]
            if track.state != "LOST":
                track.state = "LOST"
                if track in self.tracked_stracks:
                    self.tracked_stracks.remove(track)
                self.lost_stracks.append(track)

        # Step 4: Associate unconfirmed tracks with remaining high detections
        rem_dets_high = [dets_high[i] for i in unmatched_dets_a]
        matched_c, unmatched_unconfirmed, remaining_new_dets = self._associate(unconfirmed_tracks, rem_dets_high, iou_thresh=0.3)

        for t_idx, d_idx in matched_c:
            unconfirmed_tracks[t_idx].update(rem_dets_high[d_idx], ts)

        for t_idx in unmatched_unconfirmed:
            track = unconfirmed_tracks[t_idx]
            track.state = "REMOVED"
            if track in self.tracked_stracks:
                self.tracked_stracks.remove(track)
            self.removed_stracks.append(track)

        # Step 5: Initialize new tracklets from unassociated high-score detections
        for d_idx in remaining_new_dets:
            det = rem_dets_high[d_idx]
            new_track = Tracklet(
                track_id=self._next_id,
                camera_id=self.camera_id,
                t_in=ts,
                t_out=ts,
                bbox=det.bbox,
                history=[det.bbox]
            )
            self._next_id += 1
            self.tracked_stracks.append(new_track)

        # Step 6: Prune long-lost tracklets
        for track in list(self.lost_stracks):
            if track.time_since_update > self.max_time_lost:
                track.state = "REMOVED"
                self.lost_stracks.remove(track)
                self.removed_stracks.append(track)

        # Return currently active tracked targets
        return [t for t in self.tracked_stracks if t.state == "TRACKED"]

    def _associate(
        self,
        tracks: List[Tracklet],
        dets: List[Detection],
        iou_thresh: float = 0.3
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Hungarian bipartite association based on IoU distance cost matrix."""
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))

        cost_matrix = np.zeros((len(tracks), len(dets)), dtype=np.float32)
        for i, t in enumerate(tracks):
            for j, d in enumerate(dets):
                iou = compute_iou(t.bbox, d.bbox)
                cost_matrix[i, j] = 1.0 - iou

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(dets)))

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] <= (1.0 - iou_thresh):
                matched.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        return matched, list(unmatched_tracks), list(unmatched_dets)
