"""Incident Management Workflow.

Coordinates incident cases, binds probe observations and seed tracks, and maintains
case progression states.
"""

from typing import Dict, Any, List, Optional
import time
import uuid

from app.database.repository import Repository
from app.database.models import IncidentCase, PersonTarget

class IncidentManager:
    """Manages active investigations and seed target bindings."""

    def __init__(self, repo: Repository):
        self.repo = repo

    def create_incident(
        self,
        case_number: str,
        title: str,
        camera_id: str,
        incident_time: float,
        description: str = "",
        priority: str = "HIGH",
        officer_in_charge: str = "AP-EG-8821",
        seed_track_id: Optional[str] = None
    ) -> IncidentCase:
        inc_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        incident = IncidentCase(
            incident_id=inc_id,
            case_number=case_number,
            title=title,
            description=description,
            camera_id=camera_id,
            incident_time=incident_time,
            status="OPEN",
            priority=priority,
            officer_in_charge=officer_in_charge,
            seed_track_id=seed_track_id,
            created_at=time.time()
        )
        self.repo.save_incident(incident)

        # Also create initial PersonTarget if seed track provided
        if seed_track_id:
            poi = PersonTarget(
                person_id=f"POI-{uuid.uuid4().hex[:8].upper()}",
                target_code=f"POI-{case_number}",
                canonical_name="UNIDENTIFIED_TARGET",
                status="PERSON_OF_INTEREST",
                notes=f"Linked to seed incident {case_number} at {camera_id}",
                created_at=time.time(),
                updated_at=time.time()
            )
            self.repo.save_person(poi)

        self.repo.log_audit(
            action="INCIDENT_CREATED",
            details=f"Created incident {case_number} at {camera_id} by {officer_in_charge}",
            operator=officer_in_charge
        )
        return incident

    def get_incident_details(self, incident_id: str) -> Optional[Dict[str, Any]]:
        inc = self.repo.get_incident_by_id(incident_id)
        if not inc:
            return None

        data = {
            "incident_id": inc.incident_id,
            "case_number": inc.case_number,
            "title": inc.title,
            "description": inc.description,
            "camera_id": inc.camera_id,
            "incident_time": inc.incident_time,
            "status": inc.status,
            "priority": inc.priority,
            "officer_in_charge": inc.officer_in_charge,
            "seed_track_id": inc.seed_track_id,
            "created_at": inc.created_at
        }

        # Attach seed track details if exists
        if inc.seed_track_id:
            data["seed_track"] = self.repo.get_track_by_id(inc.seed_track_id)
            data["seed_feature"] = self.repo.get_feature_for_track(inc.seed_track_id)

        return data
