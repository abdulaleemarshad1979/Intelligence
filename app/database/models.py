"""Database models for CCTV Intelligence Platform."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
import time

@dataclass
class CriminalRecord:
    id: str
    fir_no: str
    unit_name: str
    subdivision: str
    police_station: str
    accused_name: str
    alias: str = ""
    age: int = 0
    gender: str = "Male"
    acts_sec: str = ""
    brief_facts: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    status_of_case: str = "Under Investigation"
    photo_url: str = ""
    
    # Biometric Reference Profile
    known_height_cm: float = 170.0
    torso_leg_ratio: float = 0.85
    stride_length_cm: float = 65.0
    posture_lean_angle: float = 4.0  # degrees
    posture_correctness: float = 0.88
    clothing_upper_color: str = "#334455"
    clothing_lower_color: str = "#112233"
    
    # Embeddings
    face_embedding: List[float] = field(default_factory=list)
    body_embedding: List[float] = field(default_factory=list)
    gait_embedding: List[float] = field(default_factory=list)

@dataclass
class TrackObservation:
    track_id: str
    camera_id: str
    first_seen: float
    last_seen: float
    frame_count: int
    best_frame_path: str = ""
    face_visible: bool = False
    face_status: str = "UNAVAILABLE"  # FULL_VISIBLE, PARTIAL_UPPER, MASKED, UNAVAILABLE
    face_tier_details: Dict[str, Any] = field(default_factory=dict)
    
    # Body & Height Biometrics
    estimated_height_cm: float = 0.0
    body_proportions: Dict[str, float] = field(default_factory=dict)
    clothing_upper: str = "#000000"
    clothing_lower: str = "#000000"
    
    # Gait Biometrics
    stride_length_px: float = 0.0
    stride_length_cm: float = 0.0
    cadence_steps_per_sec: float = 0.0
    spine_tilt_deg: float = 0.0
    posture_score: float = 0.0
    gait_wave: List[float] = field(default_factory=list)
    
    # Computed Embeddings
    face_embedding: List[float] = field(default_factory=list)
    body_embedding: List[float] = field(default_factory=list)
    gait_embedding: List[float] = field(default_factory=list)

@dataclass
class MatchEvent:
    event_id: str
    track_id: str
    camera_id: str
    suspect_id: str
    suspect_name: str
    fir_no: str
    total_confidence: float
    face_score: float
    body_score: float
    gait_score: float
    height_score: float
    is_face_available: bool
    status: str  # HIGH_CONFIDENCE, REVIEW_REQUIRED, UNKNOWN_PERSON
    evidence_breakdown: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

# ==================== GOTHAM INVESTIGATION ONTOLOGY MODELS ====================

@dataclass
class PersonTarget:
    """Canonical Person / Target of Interest in an investigation graph."""
    person_id: str
    target_code: str  # e.g., "POI-2026-088"
    canonical_name: str = "UNIDENTIFIED_TARGET"
    status: str = "PERSON_OF_INTEREST"  # UNIDENTIFIED_TARGET, PERSON_OF_INTEREST, CONFIRMED_SUSPECT, CLEARED
    notes: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

@dataclass
class CameraEntity:
    """CCTV Node in the city-wide camera topology graph."""
    camera_id: str
    name: str
    latitude: float
    longitude: float
    zone: str = "Central Division"
    view_direction: str = "NORTH"
    connected_topology: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = True
    ip_address: str = "127.0.0.1"
    rtsp_url: str = ""
    manufacturer: str = "Generic ONVIF"
    model_name: str = "IP Camera"
    mac_address: str = ""
    discovery_status: str = "APPROVED"  # DISCOVERED, APPROVED, REJECTED

@dataclass
class IncidentCase:
    """Investigative Case / Incident initiating a person tracking workflow."""
    incident_id: str
    case_number: str  # e.g., "INC-2026-0041"
    title: str
    description: str = ""
    camera_id: str = "CAM-017"
    incident_time: float = field(default_factory=time.time)
    status: str = "OPEN"  # OPEN, INVESTIGATING, ADJUDICATED, CLOSED
    priority: str = "HIGH"  # ROUTINE, MEDIUM, HIGH, CRITICAL
    officer_in_charge: str = "AP-EG-8821"
    seed_track_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)

@dataclass
class ObservationRecord:
    """Discrete geotemporal observation snapshot from a CCTV camera track."""
    observation_id: str
    track_id: str
    camera_id: str
    timestamp: float
    frame_number: int = 0
    crop_path: str = ""
    bbox: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

@dataclass
class FeatureRecord:
    """Multi-modal feature vector extraction attached to an observation/track."""
    feature_id: str
    track_id: str
    observation_id: Optional[str] = None
    face_status: str = "UNAVAILABLE"  # FULL_VISIBLE, PARTIAL_UPPER, MASKED, UNAVAILABLE
    face_embedding: List[float] = field(default_factory=list)
    body_embedding: List[float] = field(default_factory=list)
    gait_embedding: List[float] = field(default_factory=list)
    pose_landmarks: Dict[str, Any] = field(default_factory=dict)
    clothing_attributes: Dict[str, Any] = field(default_factory=dict)
    height_cm: float = 0.0
    carried_objects: List[str] = field(default_factory=list)  # e.g. ["backpack", "umbrella"]
    direction: str = "NORTH"
    created_at: float = field(default_factory=time.time)

@dataclass
class RelationshipLink:
    """Directional or associative edge between investigation objects."""
    relationship_id: str
    source_type: str  # TRACK, PERSON, INCIDENT, OBSERVATION
    source_id: str
    target_type: str  # TRACK, PERSON, INCIDENT, OBSERVATION
    target_id: str
    relationship_type: str  # CANDIDATE_SAME_PERSON, CONFIRMED_SAME_PERSON, TRANSIT_STEP, ASSOCIATE
    confidence_score: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    status: str = "CANDIDATE"  # CANDIDATE, REVIEW_PENDING, CONFIRMED, REJECTED
    created_at: float = field(default_factory=time.time)

@dataclass
class AdjudicationReview:
    """Human-in-the-loop review record for consequential link decisions."""
    review_id: str
    relationship_id: Optional[str]
    target_id: Optional[str]
    reviewer_badge: str
    reviewer_name: str
    decision: str  # CONFIRM_IDENTITY, REJECT_ASSOCIATION, DEFER_REVIEW
    review_notes: str = ""
    timestamp: float = field(default_factory=time.time)
