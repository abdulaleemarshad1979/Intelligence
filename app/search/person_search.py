"""Person Search & Candidate Discovery Coordinator.

Executes the central "FIND THIS PERSON" investigation query, taking an incident observation,
searching candidate tracks across the CCTV network, evaluating multimodal evidence,
and ranking candidate associations.
"""

from typing import Dict, Any, List, Optional
from app.database.repository import Repository
from app.search.cross_camera_search import CrossCameraSearchEngine
from app.search.temporal_search import TemporalSearchEngine
from app.association.track_association import TrackAssociator, CandidateAssociation

class PersonSearchCoordinator:
    """Orchestrates Gotham-style person investigation searches."""

    def __init__(self, repo: Repository, camera_registry: Dict[str, Any]):
        self.repo = repo
        self.camera_search = CrossCameraSearchEngine(camera_registry)
        self.temporal_search = TemporalSearchEngine()
        self.associator = TrackAssociator()

    def find_associated_tracks(
        self,
        probe_track_id: str,
        time_horizon_seconds: float = 7200.0,
        min_composite_score: float = 0.40
    ) -> List[CandidateAssociation]:
        """Search and link candidate tracks across the CCTV observation database."""
        probe_track = self.repo.get_track_by_id(probe_track_id)
        if not probe_track:
            return []

        probe_feat_rec = self.repo.get_feature_for_track(probe_track_id)
        if probe_feat_rec:
            probe_feat = {
                "face_status": probe_feat_rec.face_status,
                "face_embedding": probe_feat_rec.face_embedding,
                "body_embedding": probe_feat_rec.body_embedding,
                "gait_embedding": probe_feat_rec.gait_embedding,
                "clothing_upper": probe_feat_rec.clothing_attributes.get("upper_color", "#334455"),
                "clothing_lower": probe_feat_rec.clothing_attributes.get("lower_color", "#112233"),
                "height_cm": probe_feat_rec.height_cm,
                "carried_objects": probe_feat_rec.carried_objects,
                "direction": probe_feat_rec.direction
            }
        else:
            probe_feat = {
                "face_status": probe_track.get("face_status", "UNAVAILABLE"),
                "face_embedding": [],
                "body_embedding": [],
                "gait_embedding": [],
                "clothing_upper": probe_track.get("clothing_upper", "#334455"),
                "clothing_lower": probe_track.get("clothing_lower", "#112233"),
                "height_cm": probe_track.get("estimated_height_cm", 175.0),
                "carried_objects": ["backpack"],
                "direction": probe_track.get("direction", "NORTH")
            }

        probe_time = probe_track.get("start_time") or probe_track.get("first_seen") or 0.0
        probe_cam = probe_track.get("camera_id", "CAM-017")

        all_tracks = self.repo.get_all_tracks()
        candidate_associations = []

        for cand_track in all_tracks:
            cand_id = cand_track.get("track_id")
            if cand_id == probe_track_id:
                continue

            cand_time = cand_track.get("start_time") or cand_track.get("first_seen") or 0.0
            cand_cam = cand_track.get("camera_id", "")

            # Temporal filter: only subsequent or near-contemporaneous observations within horizon
            if not (0 <= (cand_time - probe_time) <= time_horizon_seconds):
                # Also allow tracks slightly before or within small window
                if abs(cand_time - probe_time) > time_horizon_seconds:
                    continue

            # Load candidate feature
            cand_feat_rec = self.repo.get_feature_for_track(cand_id)
            if cand_feat_rec:
                cand_feat = {
                    "face_status": cand_feat_rec.face_status,
                    "face_embedding": cand_feat_rec.face_embedding,
                    "body_embedding": cand_feat_rec.body_embedding,
                    "gait_embedding": cand_feat_rec.gait_embedding,
                    "clothing_upper": cand_feat_rec.clothing_attributes.get("upper_color", "#334455"),
                    "clothing_lower": cand_feat_rec.clothing_attributes.get("lower_color", "#112233"),
                    "height_cm": cand_feat_rec.height_cm,
                    "carried_objects": cand_feat_rec.carried_objects,
                    "direction": cand_feat_rec.direction
                }
            else:
                cand_feat = {
                    "face_status": cand_track.get("face_status", "UNAVAILABLE"),
                    "face_embedding": [],
                    "body_embedding": [],
                    "gait_embedding": [],
                    "clothing_upper": cand_track.get("clothing_upper", "#334455"),
                    "clothing_lower": cand_track.get("clothing_lower", "#112233"),
                    "height_cm": cand_track.get("estimated_height_cm", 175.0),
                    "carried_objects": ["backpack"],
                    "direction": cand_track.get("direction", "NORTH")
                }

            dist_m = self.camera_search.get_distance_meters(probe_cam, cand_cam)
            assoc = self.associator.evaluate_association(
                probe_track=probe_track,
                cand_track=cand_track,
                probe_feat=probe_feat,
                cand_feat=cand_feat,
                distance_meters=dist_m
            )

            # Keep candidate associations that meet minimum threshold
            if assoc.fusion_result.get("composite_score", 0.0) >= min_composite_score:
                candidate_associations.append(assoc)

        # Sort candidate associations chronologically and then by score
        candidate_associations.sort(key=lambda a: (a.candidate_time, -a.fusion_result.get("composite_score", 0.0)))
        return candidate_associations
