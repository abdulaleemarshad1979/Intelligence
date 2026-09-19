"""Suspicious Activity & CCTV Behavioral Analytics Module.

Analyzes trajectory vectors, velocity dynamics, dwell times, and posture shifts to detect:
1. Loitering (lingering in sensitive zones with low net displacement)
2. Sudden Sprinting / Fleeing (rapid velocity surge from walking to running)
3. Erratic Pacing / Directional Reversals (nervous pacing or scouting)
4. Crouch / Concealment Anomaly (sudden vertical shrinkage to hide from camera)
"""

import math
import time
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class BehaviorAlert:
    alert_id: str
    track_id: str
    camera_id: str
    alert_type: str  # LOITERING, SUDDEN_SPRINTING, ERRATIC_PACING, CROUCH_CONCEALMENT
    severity: str    # CRITICAL, WARNING, INFO
    confidence: float
    description: str
    timestamp: float = field(default_factory=time.time)
    metrics: Dict[str, Any] = field(default_factory=dict)

class CCTVBehaviorAnalyzer:
    """Evaluates multi-frame track trajectory and pose histories for suspicious CCTV events."""

    def __init__(
        self,
        loiter_min_dwell_sec: float = 12.0,
        loiter_max_displacement_px: float = 120.0,
        sprint_velocity_threshold_px_per_sec: float = 180.0,
        erratic_turns_threshold: int = 3
    ):
        self.loiter_min_dwell = loiter_min_dwell_sec
        self.loiter_max_displacement = loiter_max_displacement_px
        self.sprint_velocity_threshold = sprint_velocity_threshold_px_per_sec
        self.erratic_turns_threshold = erratic_turns_threshold

    def analyze_track_behavior(
        self,
        track_id: str,
        camera_id: str,
        trajectory_history: List[Dict[str, Any]],  # [{frame_id, timestamp, bbox: [x,y,w,h], posture_score, height_px}]
        fps: float = 25.0
    ) -> List[BehaviorAlert]:
        """Examine trajectory sequence and trigger contextual security alerts."""
        alerts: List[BehaviorAlert] = []
        if len(trajectory_history) < 6:
            return alerts

        timestamps = [item.get("timestamp", i / fps) for i, item in enumerate(trajectory_history)]
        bboxes = [item.get("bbox", [0, 0, 50, 100]) for item in trajectory_history]
        centroids = [(b[0] + b[2] / 2.0, b[1] + b[3] / 2.0) for b in bboxes]
        heights = [b[3] for b in bboxes]

        duration = max(0.1, timestamps[-1] - timestamps[0])

        # 1. LOITERING DETECTION
        # Total distance traveled along path vs Net distance from start to finish
        path_distance = 0.0
        for i in range(1, len(centroids)):
            dx = centroids[i][0] - centroids[i-1][0]
            dy = centroids[i][1] - centroids[i-1][1]
            path_distance += math.sqrt(dx*dx + dy*dy)

        net_dx = centroids[-1][0] - centroids[0][0]
        net_dy = centroids[-1][1] - centroids[0][1]
        net_displacement = math.sqrt(net_dx*net_dx + net_dy*net_dy)

        if duration >= self.loiter_min_dwell and net_displacement < self.loiter_max_displacement and path_distance > 40.0:
            loiter_conf = min(0.96, 0.50 + (duration / 30.0) * 0.4)
            alerts.append(BehaviorAlert(
                alert_id=f"ALT-LOITER-{track_id}-{int(time.time())}",
                track_id=track_id,
                camera_id=camera_id,
                alert_type="LOITERING",
                severity="WARNING" if duration < 25.0 else "CRITICAL",
                confidence=round(loiter_conf, 2),
                description=f"Individual lingering in sector for {duration:.1f}s with minimal net displacement ({net_displacement:.1f}px).",
                metrics={
                    "dwell_time_sec": round(duration, 1),
                    "net_displacement_px": round(net_displacement, 1),
                    "path_length_px": round(path_distance, 1)
                }
            ))

        # 2. SUDDEN SPRINTING / FLEEING DETECTION
        # Calculate instantaneous velocities between windowed steps
        step = max(1, len(centroids) // 5)
        velocities = []
        for i in range(step, len(centroids), step):
            dt = max(0.04, timestamps[i] - timestamps[i-step])
            dx = centroids[i][0] - centroids[i-step][0]
            dy = centroids[i][1] - centroids[i-step][1]
            dist = math.sqrt(dx*dx + dy*dy)
            v = dist / dt  # px per second
            velocities.append(v)

        if len(velocities) >= 2:
            max_v = max(velocities)
            min_v = min(velocities)
            accel = max_v - min_v
            if max_v > self.sprint_velocity_threshold and accel > 80.0:
                sprint_conf = min(0.95, (max_v / (self.sprint_velocity_threshold * 1.5)))
                alerts.append(BehaviorAlert(
                    alert_id=f"ALT-SPRINT-{track_id}-{int(time.time())}",
                    track_id=track_id,
                    camera_id=camera_id,
                    alert_type="SUDDEN_SPRINTING",
                    severity="CRITICAL",
                    confidence=round(sprint_conf, 2),
                    description=f"Abrupt rapid acceleration detected (peak speed: {max_v:.1f} px/s, surge: +{accel:.1f} px/s). Potential fleeing suspect.",
                    metrics={
                        "peak_velocity_px_s": round(max_v, 1),
                        "acceleration_delta": round(accel, 1)
                    }
                ))

        # 3. ERRATIC PACING / REPETITIVE DIRECTION REVERSALS
        reversals = 0
        step_rev = max(2, len(centroids) // 8)
        vecs = []
        for i in range(step_rev, len(centroids), step_rev):
            vx = centroids[i][0] - centroids[i-step_rev][0]
            vy = centroids[i][1] - centroids[i-step_rev][1]
            mag = math.sqrt(vx*vx + vy*vy)
            if mag > 10.0:
                vecs.append((vx / mag, vy / mag))

        for j in range(1, len(vecs)):
            dot = vecs[j][0]*vecs[j-1][0] + vecs[j][1]*vecs[j-1][1]
            if dot < -0.4:  # Heading reversed > 115 degrees
                reversals += 1

        if reversals >= self.erratic_turns_threshold:
            alerts.append(BehaviorAlert(
                alert_id=f"ALT-ERRATIC-{track_id}-{int(time.time())}",
                track_id=track_id,
                camera_id=camera_id,
                alert_type="ERRATIC_PACING",
                severity="WARNING",
                confidence=round(min(0.90, 0.55 + reversals * 0.1), 2),
                description=f"Repetitive directional reversals ({reversals} turnarounds). Pacing or casing perimeter.",
                metrics={
                    "reversals_count": reversals,
                    "headings_tracked": len(vecs)
                }
            ))

        # 4. CROUCH / SUDDEN CONCEALMENT
        if len(heights) >= 4:
            initial_h = np.mean(heights[:2])
            final_h = heights[-1]
            if initial_h > 80 and final_h < initial_h * 0.55:
                alerts.append(BehaviorAlert(
                    alert_id=f"ALT-CROUCH-{track_id}-{int(time.time())}",
                    track_id=track_id,
                    camera_id=camera_id,
                    alert_type="CROUCH_CONCEALMENT",
                    severity="WARNING",
                    confidence=0.82,
                    description=f"Sudden 45%+ bounding box vertical collapse ({initial_h:.0f}px -> {final_h:.0f}px). Subject crouched or dropped.",
                    metrics={
                        "initial_height": round(initial_h, 1),
                        "final_height": round(final_h, 1)
                    }
                ))

        return alerts
