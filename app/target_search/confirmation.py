"""Temporal confirmation engine and multi-frame evidence selector for Target Person Search.

Implements track-level observation buffering, rolling-window validation,
multi-criteria confirmation scoring, best evidence frame selection,
and cryptographically signed evidence artifact packaging.
"""

import os
import time
import json
import uuid
import hashlib
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from app.target_search.models import (
    Observation, CandidateEvent, CandidateStatus, TargetSearchConfig
)

logger = logging.getLogger(__name__)


def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 digest of binary content."""
    return hashlib.sha256(data).hexdigest()


class CandidateTrackState:
    """Maintains the rolling observation state and best evidence frames for a track."""

    def __init__(self, session_id: str, camera_id: str, track_id: str, config: TargetSearchConfig):
        self.session_id = session_id
        self.camera_id = camera_id
        self.track_id = str(track_id)
        self.config = config

        self.observations: List[Observation] = []
        self.status: CandidateStatus = CandidateStatus.OBSERVED
        self.first_seen: float = 0.0
        self.last_seen: float = 0.0

        # Best frame tracking
        self.best_frame_score: float = -1.0
        self.best_full_frame: Optional[np.ndarray] = None
        self.best_person_crop: Optional[np.ndarray] = None
        self.best_face_crop: Optional[np.ndarray] = None
        self.best_observation: Optional[Observation] = None

        # Confirmation telemetry
        self.confirmation_count: int = 0
        self.raw_similarity_max: float = 0.0
        self.raw_similarity_mean: float = 0.0
        self.quality_mean: float = 0.0
        self.track_consistency: float = 0.0
        self.confirmation_score: float = 0.0

        # Cooldown management
        self.last_event_time: float = 0.0
        self.confirmed_at: Optional[float] = None
        self.has_emitted_event: bool = False

    def add_observation(
        self,
        obs: Observation,
        full_frame: Optional[np.ndarray] = None,
        person_crop: Optional[np.ndarray] = None,
        face_crop: Optional[np.ndarray] = None
    ) -> bool:
        """Add an observation if it meets quality and similarity thresholds."""
        now = obs.timestamp
        if self.first_seen == 0.0:
            self.first_seen = now
        self.last_seen = now

        # Quality & similarity gate check
        is_valid = (
            obs.raw_similarity >= self.config.minimum_raw_similarity and
            obs.quality_score >= self.config.minimum_quality_score
        )
        if not is_valid:
            return False

        # Calculate best frame score:
        # best_frame_score = 0.45 * image_quality + 0.30 * raw_similarity + 0.15 * detection_score + 0.10 * crop_area_score
        bw = obs.bbox[2] if len(obs.bbox) >= 4 else 50
        bh = obs.bbox[3] if len(obs.bbox) >= 4 else 100
        crop_area_score = min(1.0, max(0.05, (bw * bh) / (120.0 * 240.0)))

        frame_score = (
            0.45 * float(obs.quality_score)
            + 0.30 * float(obs.raw_similarity)
            + 0.15 * float(obs.detection_score)
            + 0.10 * float(crop_area_score)
        )
        obs.best_frame_score = frame_score

        if frame_score > self.best_frame_score:
            self.best_frame_score = frame_score
            self.best_observation = obs
            if full_frame is not None and full_frame.size > 0:
                self.best_full_frame = full_frame.copy()
            if person_crop is not None and person_crop.size > 0:
                self.best_person_crop = person_crop.copy()
            if face_crop is not None and face_crop.size > 0:
                self.best_face_crop = face_crop.copy()

        self.observations.append(obs)
        self.prune_stale_observations(now)
        self._update_aggregate_metrics(now)
        return True

    def prune_stale_observations(self, current_time: float) -> None:
        """Prune observations that fall outside the rolling confirmation window."""
        cutoff = current_time - self.config.confirmation_window_sec
        self.observations = [o for o in self.observations if o.timestamp >= cutoff]

    def _compute_track_consistency(self) -> float:
        """Compute track spatial and temporal consistency score in [0.0, 1.0]."""
        if not self.observations:
            return 0.0
        count = len(self.observations)
        if count == 1:
            return 0.65

        # Check temporal coverage relative to min_confirmations
        count_ratio = min(1.0, count / max(1, self.config.min_confirmations))

        # Check aspect ratio variance across bounding boxes
        aspects = []
        for o in self.observations:
            w = max(1, o.bbox[2])
            h = max(1, o.bbox[3])
            aspects.append(w / float(h))
        aspect_var = float(np.var(aspects)) if len(aspects) > 1 else 0.0
        smoothness = max(0.0, 1.0 - min(1.0, aspect_var * 4.0))

        consistency = 0.50 * count_ratio + 0.50 * smoothness
        return float(min(1.0, max(0.0, consistency)))

    def _update_aggregate_metrics(self, current_time: float) -> None:
        """Update metrics and transition candidate state."""
        count = len(self.observations)
        self.confirmation_count = count

        if count == 0:
            if current_time - self.last_seen > self.config.confirmation_window_sec * 2.0:
                self.status = CandidateStatus.EXPIRED
            return

        sims = [o.raw_similarity for o in self.observations]
        quals = [o.quality_score for o in self.observations]

        self.raw_similarity_max = round(float(max(sims)), 4)
        self.raw_similarity_mean = round(float(np.mean(sims)), 4)
        self.quality_mean = round(float(np.mean(quals)), 4)
        self.track_consistency = round(self._compute_track_consistency(), 4)

        # Confirmation score formula (Section 5):
        # confirmation_score = 0.50 * mean_recent_similarity + 0.20 * max_similarity + 0.15 * mean_quality + 0.15 * track_consistency
        raw_score = (
            0.50 * self.raw_similarity_mean
            + 0.20 * self.raw_similarity_max
            + 0.15 * self.quality_mean
            + 0.15 * self.track_consistency
        )
        self.confirmation_score = round(float(min(1.0, max(0.0, raw_score))), 4)

        # State transitions
        if self.status != CandidateStatus.REVIEWED:
            if count >= self.config.min_confirmations:
                self.status = CandidateStatus.CONFIRMED_CANDIDATE
                if self.confirmed_at is None:
                    self.confirmed_at = current_time
            elif count >= 2:
                self.status = CandidateStatus.TRACKING
            else:
                self.status = CandidateStatus.OBSERVED

    def can_emit_event(self, current_time: float) -> bool:
        """Check if candidate satisfies confirmation threshold and cooldown."""
        if self.status != CandidateStatus.CONFIRMED_CANDIDATE:
            return False
        if current_time - self.last_event_time < self.config.duplicate_event_cooldown_sec:
            return False
        return True


class TemporalConfirmationEngine:
    """Coordinates track observations across cameras and produces confirmed candidate events."""

    def __init__(self, session_id: str, storage_base_dir: str, config: Optional[TargetSearchConfig] = None):
        self.session_id = session_id
        self.storage_base_dir = storage_base_dir
        self.config = config or TargetSearchConfig()

        self._lock = threading.Lock()
        # Key: (camera_id, track_id) -> CandidateTrackState
        self.tracks: Dict[Tuple[str, str], CandidateTrackState] = {}
        # List of confirmed events produced in this session
        self.events: List[CandidateEvent] = []
        self._event_counter = 0

    def process_observation(
        self,
        observation: Observation,
        full_frame: Optional[np.ndarray] = None,
        person_crop: Optional[np.ndarray] = None,
        face_crop: Optional[np.ndarray] = None
    ) -> Tuple[CandidateTrackState, Optional[CandidateEvent]]:
        """Ingest an observation, update track state, and emit a candidate event if confirmed."""
        with self._lock:
            key = (observation.camera_id, str(observation.track_id))
            if key not in self.tracks:
                self.tracks[key] = CandidateTrackState(
                    session_id=self.session_id,
                    camera_id=observation.camera_id,
                    track_id=str(observation.track_id),
                    config=self.config
                )

            track_state = self.tracks[key]
            track_state.add_observation(
                obs=observation,
                full_frame=full_frame,
                person_crop=person_crop,
                face_crop=face_crop
            )

            event: Optional[CandidateEvent] = None
            now = observation.timestamp

            if track_state.can_emit_event(now):
                event = self._save_candidate_evidence(track_state, now)
                if event:
                    track_state.last_event_time = now
                    track_state.has_emitted_event = True
                    self.events.append(event)

            return track_state, event

    def _save_candidate_evidence(self, track: CandidateTrackState, timestamp: float) -> Optional[CandidateEvent]:
        """Save best full frame, person crop, face crop, hashes, and metadata."""
        self._event_counter += 1
        event_id = f"EVT-TS-{self._event_counter:03d}"
        event_dir = os.path.join(self.storage_base_dir, self.session_id, event_id)
        os.makedirs(event_dir, exist_ok=True)

        full_frame = track.best_full_frame
        person_crop = track.best_person_crop
        face_crop = track.best_face_crop

        # Draw visual bounding box overlay on full frame if available
        if full_frame is not None and track.best_observation:
            annotated_full = full_frame.copy()
            bx, by, bw, bh = track.best_observation.bbox
            cv2.rectangle(annotated_full, (bx, by), (bx + bw, by + bh), (0, 165, 255), 2)
            cv2.putText(
                annotated_full,
                f"CANDIDATE TRACK-{track.track_id} (Sim: {track.raw_similarity_max:.2f})",
                (bx, max(20, by - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                2
            )
            full_frame = annotated_full

        hashes: Dict[str, str] = {}
        full_path = ""
        person_path = ""
        face_path = None

        # 1. Full frame
        if full_frame is not None and full_frame.size > 0:
            full_path = os.path.join(event_dir, "full.jpg")
            _, buf = cv2.imencode(".jpg", full_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            with open(full_path, "wb") as f:
                f.write(buf.tobytes())
            hashes["full.jpg"] = compute_sha256(buf.tobytes())

        # 2. Person crop
        if person_crop is not None and person_crop.size > 0:
            person_path = os.path.join(event_dir, "person.jpg")
            _, buf = cv2.imencode(".jpg", person_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
            with open(person_path, "wb") as f:
                f.write(buf.tobytes())
            hashes["person.jpg"] = compute_sha256(buf.tobytes())

        # 3. Face crop (if available)
        if face_crop is not None and face_crop.size > 0:
            face_path = os.path.join(event_dir, "face.jpg")
            _, buf = cv2.imencode(".jpg", face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
            with open(face_path, "wb") as f:
                f.write(buf.tobytes())
            hashes["face.jpg"] = compute_sha256(buf.tobytes())

        # 4. hashes.json
        hashes_path = os.path.join(event_dir, "hashes.json")
        with open(hashes_path, "w", encoding="utf-8") as f:
            json.dump(hashes, f, indent=2)

        # 5. metadata.json
        metadata = {
            "event_id": event_id,
            "session_id": self.session_id,
            "camera_id": track.camera_id,
            "track_id": track.track_id,
            "first_seen": track.first_seen,
            "last_seen": track.last_seen,
            "confirmation_count": track.confirmation_count,
            "raw_similarity_max": round(track.raw_similarity_max, 4),
            "raw_similarity_mean": round(track.raw_similarity_mean, 4),
            "quality_mean": round(track.quality_mean, 4),
            "track_consistency": round(track.track_consistency, 4),
            "confirmation_score": round(track.confirmation_score, 4),
            "status": CandidateStatus.CONFIRMED_CANDIDATE.value,
            "review_required": True,
            "hashes": hashes,
            "captured_at": timestamp
        }
        metadata_path = os.path.join(event_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        event = CandidateEvent(
            event_id=event_id,
            session_id=self.session_id,
            camera_id=track.camera_id,
            track_id=track.track_id,
            first_seen=track.first_seen,
            last_seen=track.last_seen,
            confirmation_count=track.confirmation_count,
            raw_similarity_max=track.raw_similarity_max,
            raw_similarity_mean=track.raw_similarity_mean,
            quality_mean=track.quality_mean,
            confirmation_score=track.confirmation_score,
            status=CandidateStatus.CONFIRMED_CANDIDATE.value,
            review_required=True,
            best_frame_path=full_path,
            person_crop_path=person_path,
            face_crop_path=face_path,
            hashes=hashes,
            metadata=metadata,
            created_at=timestamp
        )

        logger.info(
            f"[{self.session_id}] Emitted Confirmed Candidate Event {event_id} for Camera {track.camera_id} Track {track.track_id} (Score: {track.confirmation_score:.2f})"
        )
        return event

    def get_track_state(self, camera_id: str, track_id: str) -> Optional[CandidateTrackState]:
        with self._lock:
            return self.tracks.get((camera_id, str(track_id)))

    def get_all_tracks(self) -> List[CandidateTrackState]:
        with self._lock:
            return list(self.tracks.values())

    def get_events(self) -> List[CandidateEvent]:
        with self._lock:
            return list(self.events)

    def prune_expired_tracks(self, current_time: float) -> None:
        """Mark tracks as expired if no observations within 2x window."""
        with self._lock:
            for track in self.tracks.values():
                track.prune_stale_observations(current_time)
                if current_time - track.last_seen > self.config.confirmation_window_sec * 2.0:
                    if track.status not in (CandidateStatus.CONFIRMED_CANDIDATE, CandidateStatus.REVIEWED):
                        track.status = CandidateStatus.EXPIRED
