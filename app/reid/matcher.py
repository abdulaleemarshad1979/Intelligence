"""Re-ID candidate matcher searching the gallery and generating ranked evidence profiles."""

import uuid
from typing import List, Dict, Any, Optional
from app.database.models import TrackObservation, MatchEvent
from app.reid.gallery import SuspectGallery
from app.fusion.evidence import EvidenceFusionEngine
from app.database.repository import Repository

class CandidateMatcher:
    def __init__(self, gallery: Optional[SuspectGallery] = None, repository: Optional[Repository] = None):
        self.repo = repository or Repository()
        self.gallery = gallery or SuspectGallery(self.repo)
        self.engine = EvidenceFusionEngine()

    def match_track(self, track: TrackObservation, top_k: int = 5) -> List[Dict[str, Any]]:
        """Match a track observation against the entire criminal gallery and return top candidates."""
        suspects = self.gallery.get_all()
        candidates = []

        for s in suspects:
            ev = self.engine.evaluate_candidate(track, s)
            candidates.append(ev)

        # Sort descending by total confidence
        candidates.sort(key=lambda x: x["total_confidence"], reverse=True)
        top_candidates = candidates[:top_k]

        # If highest candidate exceeds review threshold, log match event
        if top_candidates and top_candidates[0]["total_confidence"] >= 0.45:
            best = top_candidates[0]
            event = MatchEvent(
                event_id=f"EVT-{uuid.uuid4().hex[:8].upper()}",
                track_id=track.track_id,
                camera_id=track.camera_id,
                suspect_id=best["suspect_id"],
                suspect_name=best["suspect_name"],
                fir_no=best["fir_no"],
                total_confidence=best["total_confidence"],
                face_score=best["scores"]["face_score"],
                body_score=best["scores"]["body_score"],
                gait_score=best["scores"]["gait_score"],
                height_score=best["scores"]["height_score"],
                is_face_available=best["is_face_available"],
                status=best["status"],
                evidence_breakdown=best
            )
            self.repo.save_match_event(event)

        return top_candidates
