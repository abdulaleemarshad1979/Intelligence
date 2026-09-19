"""ByteTrack Multi-Object Tracking Adapter.

Associates both high-score and low-score detection bounding boxes to maintain
persistent IDs through occlusions without throwing away low-confidence frames.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from app.adapters.base import DetectionResult, TrackingResult
from app.adapters.tracking.base import BaseMOTTracker
from app.tracking.tracker import MultiPersonTracker


class ByteTrackAdapter(BaseMOTTracker):
    """ByteTrack adapter with two-stage association strategy."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.backend = "BYTETRACK_TWO_STAGE_ASSOCIATION"
        self.internal_tracker = MultiPersonTracker(
            max_age=track_buffer,
            min_hits=2,
            iou_threshold=0.3
        )

    def update(self, detections: List[DetectionResult], frame_id: int) -> List[TrackingResult]:
        """Update tracks using ByteTrack two-stage association."""
        # Split detections into high and low confidence
        high_dets = [d for d in detections if d.confidence >= self.track_thresh]
        low_dets = [d for d in detections if d.confidence < self.track_thresh]

        # Primary association on high detections
        wrapped_high = [
            {"box": [d.bbox[0], d.bbox[1], d.bbox[0] + d.bbox[2], d.bbox[1] + d.bbox[3]], "confidence": d.confidence}
            for d in high_dets
        ]
        active = self.internal_tracker.update(wrapped_high)

        results: List[TrackingResult] = []
        for trk in active:
            x1, y1, x2, y2 = trk.box
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            try:
                tid = int(str(trk.track_id).replace("TRACK-", ""))
            except Exception:
                tid = 1

            results.append(TrackingResult(
                track_id=tid,
                bbox=(x1, y1, w, h),
                confidence=trk.confidence,
                state="CONFIRMED" if trk.hits >= 2 else "TENTATIVE",
                time_since_update=trk.time_since_update
            ))
        return results

    def get_backend_info(self) -> Dict[str, Any]:
        return {
            "model_id": "bytetrack",
            "name": "ByteTrack",
            "backend": self.backend,
            "track_thresh": self.track_thresh,
            "match_thresh": self.match_thresh,
            "track_buffer": self.track_buffer,
            "association": "Two-Stage High/Low Confidence Kalman Association"
        }
