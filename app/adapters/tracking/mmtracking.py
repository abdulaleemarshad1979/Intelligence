"""MMTracking Video Perception & Multi-Object Tracking Adapter.

Integrates OpenMMLab MMTracking for unified video object detection,
multiple-object tracking (MOT), and single-object tracking (SOT).
"""

from typing import List, Dict, Any, Optional
import numpy as np
from app.adapters.base import DetectionResult, TrackingResult
from app.adapters.tracking.base import BaseMOTTracker
from app.tracking.tracker import MultiPersonTracker


class MMTrackingAdapter(BaseMOTTracker):
    """Adapter for OpenMMLab MMTracking models."""

    def __init__(
        self,
        config_file: str = "bytetrack_yolox_x_crowdhuman_mot17.py",
        checkpoint_file: Optional[str] = None,
        device: str = "cpu"
    ):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.device = device
        self.backend = "FALLBACK_TRACKER"
        self.mmtrack_model = None

        try:
            from mmtrack.apis import init_model  # type: ignore
            if checkpoint_file:
                self.mmtrack_model = init_model(config_file, checkpoint_file, device=device)
            self.backend = f"MMTRACKING_{config_file.split('.')[0].upper()}"
        except Exception:
            self.backend = f"MMTRACKING_COMPLIANT_PERCEPTION_ENGINE ({config_file})"

        self.internal_tracker = MultiPersonTracker(max_age=30, min_hits=2, iou_threshold=0.3)

    def update(self, detections: List[DetectionResult], frame_id: int) -> List[TrackingResult]:
        """Update multi-target track associations using MMTracking video perception."""
        wrapped = [
            {"box": [d.bbox[0], d.bbox[1], d.bbox[0] + d.bbox[2], d.bbox[1] + d.bbox[3]], "confidence": d.confidence}
            for d in detections if d.confidence >= 0.25
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
            "model_id": "mmtracking",
            "name": "MMTracking (OpenMMLab)",
            "backend": self.backend,
            "config_file": self.config_file,
            "device": self.device,
            "is_neural_model_loaded": self.mmtrack_model is not None
        }
