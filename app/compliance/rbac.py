"""Role-Based Access Control (RBAC) & Authorization for Police Surveillance Systems."""

from enum import Enum
from typing import Set, Dict, Any, Optional
from pydantic import BaseModel

class UserRole(str, Enum):
    ADMIN = "ADMIN"
    INVESTIGATOR = "INVESTIGATOR"
    FORENSIC_ANALYST = "FORENSIC_ANALYST"
    PATROL_OFFICER = "PATROL_OFFICER"
    AUDITOR = "AUDITOR"

class Permission(str, Enum):
    VIEW_LIVE_FEEDS = "VIEW_LIVE_FEEDS"
    VIEW_ALERTS = "VIEW_ALERTS"
    SEARCH_GALLERY = "SEARCH_GALLERY"
    INSPECT_DOSSIER = "INSPECT_DOSSIER"
    SIGN_OFF_MATCH = "SIGN_OFF_MATCH"
    EXPORT_EVIDENCE = "EXPORT_EVIDENCE"
    MANAGE_CALIBRATION = "MANAGE_CALIBRATION"
    MANAGE_RETENTION = "MANAGE_RETENTION"
    VIEW_AUDIT_LOGS = "VIEW_AUDIT_LOGS"

ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.ADMIN: {
        Permission.VIEW_LIVE_FEEDS,
        Permission.VIEW_ALERTS,
        Permission.SEARCH_GALLERY,
        Permission.INSPECT_DOSSIER,
        Permission.SIGN_OFF_MATCH,
        Permission.EXPORT_EVIDENCE,
        Permission.MANAGE_CALIBRATION,
        Permission.MANAGE_RETENTION,
        Permission.VIEW_AUDIT_LOGS,
    },
    UserRole.INVESTIGATOR: {
        Permission.VIEW_LIVE_FEEDS,
        Permission.VIEW_ALERTS,
        Permission.SEARCH_GALLERY,
        Permission.INSPECT_DOSSIER,
        Permission.SIGN_OFF_MATCH,
        Permission.EXPORT_EVIDENCE,
        Permission.VIEW_AUDIT_LOGS,
    },
    UserRole.FORENSIC_ANALYST: {
        Permission.VIEW_LIVE_FEEDS,
        Permission.VIEW_ALERTS,
        Permission.SEARCH_GALLERY,
        Permission.INSPECT_DOSSIER,
        Permission.EXPORT_EVIDENCE,
    },
    UserRole.PATROL_OFFICER: {
        Permission.VIEW_LIVE_FEEDS,
        Permission.VIEW_ALERTS,
    },
    UserRole.AUDITOR: {
        Permission.VIEW_AUDIT_LOGS,
        Permission.VIEW_ALERTS,
    }
}

class OfficerIdentity(BaseModel):
    user_id: str
    badge_number: str
    full_name: str
    rank: str = "Sub-Inspector"
    police_station: str = "II Town PS, Kakinada"
    role: UserRole = UserRole.INVESTIGATOR

    def can(self, permission: Permission) -> bool:
        allowed = ROLE_PERMISSIONS.get(self.role, set())
        return permission in allowed

# Default fallback context for development & live dashboard demo
DEFAULT_OFFICER = OfficerIdentity(
    user_id="OFF-7842",
    badge_number="AP-KKD-1042",
    full_name="V. R. Sekhar",
    rank="Inspector of Police",
    police_station="II Town Police Station, Kakinada",
    role=UserRole.INVESTIGATOR
)
