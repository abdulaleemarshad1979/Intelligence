"""Multi-object tracker for persistent CCTV person identity tracking."""

import numpy as np
from typing import List, Dict, Any, Optional

def compute_iou(boxA: List[int], boxB: List[int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0
    return interArea / union

class TrackItem:
    def __init__(self, track_id: str, box: List[int], confidence: float):
        self.track_id = track_id
        self.box = box  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.history: List[List[int]] = [box]
        self.confirmed = False

    @property
    def centroid(self) -> Tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    def update(self, box: List[int], confidence: float):
        self.box = box
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.history.append(box)
        if len(self.history) > 100:
            self.history.pop(0)
        if self.hits >= 3:
            self.confirmed = True

    def mark_missed(self):
        self.age += 1
        self.time_since_update += 1

class MultiPersonTracker:
    def __init__(self, max_age: int = 25, min_hits: int = 3, iou_threshold: float = 0.25):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[TrackItem] = []
        self.next_id = 1

    def update(self, detections: List[Dict[str, Any]]) -> List[TrackItem]:
        """Update tracker with frame detections. Returns currently active tracks."""
        det_boxes = [d["box"] for d in detections]
        det_confs = [d["confidence"] for d in detections]

        matched_tracks = set()
        matched_dets = set()

        if len(self.tracks) > 0 and len(det_boxes) > 0:
            # Calculate cost matrix based on IoU and centroid distance
            iou_matrix = np.zeros((len(self.tracks), len(det_boxes)), dtype=float)
            for t_idx, track in enumerate(self.tracks):
                for d_idx, dbox in enumerate(det_boxes):
                    iou = compute_iou(track.box, dbox)
                    # Add centroid proximity boost
                    tc = track.centroid
                    dc = ((dbox[0] + dbox[2]) / 2.0, (dbox[1] + dbox[3]) / 2.0)
                    dist = np.sqrt((tc[0] - dc[0])**2 + (tc[1] - dc[1])**2)
                    dist_score = max(0.0, 1.0 - (dist / 150.0))
                    
                    iou_matrix[t_idx, d_idx] = 0.6 * iou + 0.4 * dist_score

            # Greedy bipartite matching
            while True:
                max_val = np.max(iou_matrix)
                if max_val < self.iou_threshold:
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)
                self.tracks[t_idx].update(det_boxes[d_idx], det_confs[d_idx])
                
                # Zero out row and col
                iou_matrix[t_idx, :] = -1
                iou_matrix[:, d_idx] = -1

        # Handle unmatched existing tracks
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.mark_missed()

        # Handle unmatched detections (create new tracks)
        for d_idx, dbox in enumerate(det_boxes):
            if d_idx not in matched_dets:
                tid = f"TRACK-{self.next_id:04d}"
                self.next_id += 1
                new_track = TrackItem(tid, dbox, det_confs[d_idx])
                self.tracks.append(new_track)

        # Prune dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return confirmed tracks or tracks that have hits >= min_hits
        return [t for t in self.tracks if t.confirmed or t.hits >= self.min_hits]
