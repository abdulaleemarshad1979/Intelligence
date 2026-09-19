"""Track Association Engine.

Constructs and evaluates candidate associations between track observations across CCTV cameras,
validating physical transit feasibility and synthesizing transparent multi-modal evidence.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import time

from app.evidence.face import FaceEvidenceEvaluator
from app.evidence.body import BodyEvidenceEvaluator
from app.evidence.gait import GaitEvidenceEvaluator
from app.evidence.pose import PoseEvidenceEvaluator
from app.evidence.clothing import ClothingEvidenceEvaluator
from app.evidence.height import HeightEvidenceEvaluator
from app.evidence.object_features import CarriedObjectEvidenceEvaluator
from app.association.evidence_fusion import EvidenceFusionEngine
from app.association.confidence import format_evidence_narrative

@dataclass
class CandidateAssociation:
    """Evaluated association between a probe track and candidate track."""
    probe_track_id: str
    candidate_track_id: str
    probe_camera: str
    candidate_camera: str
    probe_time: float
    candidate_time: float
    time_delta_sec: float
    distance_meters: float
    transit_speed_mps: float
    is_physically_feasible: bool
    evidence_breakdown: Dict[str, Any]
    fusion_result: Dict[str, Any]
    qualitative_narrative: str
    association_status: str = "CANDIDATE"  # CANDIDATE, REVIEW_PENDING, CONFIRMED, REJECTED
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_track_id": self.probe_track_id,
            "candidate_track_id": self.candidate_track_id,
            "probe_camera": self.probe_camera,
            "candidate_camera": self.candidate_camera,
            "probe_time": self.probe_time,
            "candidate_time": self.candidate_time,
            "time_delta_sec": round(self.time_delta_sec, 1),
            "distance_meters": round(self.distance_meters, 1),
            "transit_speed_mps": round(self.transit_speed_mps, 2),
            "is_physically_feasible": self.is_physically_feasible,
            "evidence_breakdown": self.evidence_breakdown,
            "composite_score": self.fusion_result.get("composite_score", 0.0),
            "qualitative_narrative": self.qualitative_narrative,
            "association_status": self.association_status,
            "created_at": self.created_at
        }

class TrackAssociator:
    """Evaluates multimodal association between CCTV tracks."""

    def __init__(self):
        self.face_eval = FaceEvidenceEvaluator()
        self.body_eval = BodyEvidenceEvaluator()
        self.gait_eval = GaitEvidenceEvaluator()
        self.pose_eval = PoseEvidenceEvaluator()
        self.clothing_eval = ClothingEvidenceEvaluator()
        self.height_eval = HeightEvidenceEvaluator()
        self.object_eval = CarriedObjectEvidenceEvaluator()
        self.fusion = EvidenceFusionEngine()

    def evaluate_association(
        self,
        probe_track: Dict[str, Any],
        cand_track: Dict[str, Any],
        probe_feat: Dict[str, Any],
        cand_feat: Dict[str, Any],
        distance_meters: float = 400.0
    ) -> CandidateAssociation:
        probe_t = probe_track.get("start_time") or probe_track.get("first_seen") or 0.0
        cand_t = cand_track.get("start_time") or cand_track.get("first_seen") or 0.0
        dt = abs(cand_t - probe_t)

        speed_mps = (distance_meters / dt) if dt > 0 else 0.0
        # Human walking speed ~0.5 to 2.5 m/s, running up to 6.0 m/s
        is_feasible = (0.3 <= speed_mps <= 6.5) if dt > 10 else True

        # Multimodal evidence evaluations
        ev_face = self.face_eval.evaluate(probe_feat, cand_feat)
        ev_body = self.body_eval.evaluate(probe_feat, cand_feat)
        ev_gait = self.gait_eval.evaluate(probe_feat, cand_feat)
        ev_pose = self.pose_eval.evaluate(probe_feat, cand_feat)
        ev_clothing = self.clothing_eval.evaluate(probe_feat, cand_feat)
        ev_height = self.height_eval.evaluate(probe_feat, cand_feat)
        ev_object = self.object_eval.evaluate(probe_feat, cand_feat)

        travel_summary = f"{dt/60.0:.1f} mins for {distance_meters:.0f}m ({speed_mps:.1f} m/s)"
        ev_travel = {
            "is_feasible": is_feasible,
            "distance_m": distance_meters,
            "delta_t_sec": dt,
            "speed_mps": speed_mps,
            "summary": travel_summary
        }

        traj_consistent = (probe_feat.get("direction", "NORTH") == cand_feat.get("direction", "NORTH"))
        ev_trajectory = {
            "consistent": traj_consistent,
            "direction": cand_feat.get("direction", "NORTH")
        }

        evidence_breakdown = {
            "face": ev_face,
            "body": ev_body,
            "gait": ev_gait,
            "pose": ev_pose,
            "clothing": ev_clothing,
            "height": ev_height,
            "carried_objects": ev_object,
            "travel_time": ev_travel,
            "trajectory": ev_trajectory
        }

        fusion_result = self.fusion.fuse(evidence_breakdown)
        narrative = format_evidence_narrative(evidence_breakdown)

        return CandidateAssociation(
            probe_track_id=str(probe_track.get("track_id", "")),
            candidate_track_id=str(cand_track.get("track_id", "")),
            probe_camera=str(probe_track.get("camera_id", "")),
            candidate_camera=str(cand_track.get("camera_id", "")),
            probe_time=probe_t,
            candidate_time=cand_t,
            time_delta_sec=dt,
            distance_meters=distance_meters,
            transit_speed_mps=speed_mps,
            is_physically_feasible=is_feasible,
            evidence_breakdown=evidence_breakdown,
            fusion_result=fusion_result,
            qualitative_narrative=narrative
        )
