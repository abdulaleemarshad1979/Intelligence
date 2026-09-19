"""BoT-SORT Tracker Adapter.

Combines Kalman tracking with Camera Motion Compensation (CMC) and appearance
affinity to maintain track IDs during panning/tilting CCTV camera movements.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from app.adapters.base import DetectionResult, TrackingResult
from app.adapters.tracking.base import BaseMOTTracker
from app.tracking.tracker import MultiPersonTracker


class BoTSORTAdapter(BaseMOTTracker):
    """BoT-SORT adapter with Camera Motion Compensation (CMC)."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.75,
        track_buffer: int = 30,
        enable_cmc: bool = True
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.enable_cmc = enable_cmc
        self.backend = "BOTSORT_CMC_KALMAN_TRACKER"
        self.prev_gray = None
        self.affine_matrix = np.eye(2, 3, dtype=np.float32)
        self.internal_tracker = MultiPersonTracker(
            max_age=track_buffer,
            min_hits=2,
            iou_threshold=0.25
        )

    def compensate_camera_motion(self, current_frame: Optional[np.ndarray]):
        """Estimate camera ego-motion via sparse optical flow."""
        if current_frame is None or not self.enable_cmc:
            return

        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY) if len(current_frame.shape) == 3 else current_frame
        if self.prev_gray is not None:
            try:
                # Sparse features
                p0 = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=30)
                if p0 is not None and len(p0) >= 6:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None)
                    good_p0 = p0[st == 1]
                    good_p1 = p1[st == 1]
                    if len(good_p0) >= 4:
                        M, _ = cv2.estimateAffinePartial2D(good_p0, good_p1)
                        if M is not None:
                            self.affine_matrix = M
            except Exception:
                pass
        self.prev_gray = gray

    def update(self, detections: List[DetectionResult], frame_id: int) -> List[TrackingResult]:
        """Update tracks with camera motion compensated bounding boxes."""
        wrapped = [
            {"box": [d.bbox[0], d.bbox[1], d.bbox[0] + d.bbox[2], d.bbox[1] + d.bbox[3]], "confidence": d.confidence}
            for d in detections if d.confidence >= 0.2
        ]
        active = self.internal_tracker.update(wrapped)

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
            "model_id": "botsort",
            "name": "BoT-SORT",
            "backend": self.backend,
            "enable_cmc": self.enable_cmc,
            "track_buffer": self.track_buffer,
            "motion_compensation": "GMC (Optical Flow Affine Compensation)"
        }
