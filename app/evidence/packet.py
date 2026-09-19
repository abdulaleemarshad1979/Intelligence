"""Person Evidence Packet Module.

Core investigation entity encapsulating multi-modal observations (Face, Partial Face,
Body Re-ID, Clothing, Pose sequence, Gait dynamics, Calibrated Stature, Carried Objects,
Trajectory) with independent quality metrics so face is never a mandatory gatekeeper.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import time


@dataclass
class PersonEvidencePacket:
    """Central multi-modal evidence container for investigative person search."""
    track_id: str = ""
    camera_id: str = ""
    timestamp_start: float = field(default_factory=time.time)
    timestamp_end: float = field(default_factory=time.time)

    # Modalities
    face: Dict[str, Any] = field(default_factory=dict)
    partial_face: Dict[str, Any] = field(default_factory=dict)
    body_embedding: List[float] = field(default_factory=list)
    clothing: Dict[str, Any] = field(default_factory=dict)
    pose_sequence: List[Dict[str, Any]] = field(default_factory=list)
    gait_embedding: List[float] = field(default_factory=list)
    height_estimate: float = 0.0
    object_attributes: List[str] = field(default_factory=list)
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)  # (x, y, timestamp)

    # Quality scores per modality (0.0 to 1.0)
    quality: Dict[str, float] = field(default_factory=lambda: {
        "face": 0.0,
        "body": 0.0,
        "gait": 0.0,
        "pose": 0.0,
        "height": 0.0,
        "trajectory": 0.0,
        "clothing": 0.0
    })

    # Observational context
    direction_of_travel: str = "UNKNOWN"
    notes: str = ""

    def is_face_available(self) -> bool:
        """Check if face evidence is available and meets minimum quality."""
        return bool(self.face.get("is_available", False) and self.quality.get("face", 0.0) >= 0.25)

    def is_partial_face_available(self) -> bool:
        """Check if upper/mid tiers are visible even if lower face is occluded/masked."""
        tiers = self.partial_face.get("tiers", {})
        return bool(tiers.get("upper") or tiers.get("mid"))

    def get_available_modalities(self) -> List[str]:
        """Return list of active, populated modalities (Face never gatekeeps others!)."""
        mods = []
        if self.is_face_available():
            mods.append("face")
        if self.body_embedding and self.quality.get("body", 0.0) > 0.1:
            mods.append("body")
        if self.clothing and (self.clothing.get("clothing_upper") or self.clothing.get("upper")):
            mods.append("clothing")
        if self.pose_sequence or self.quality.get("pose", 0.0) > 0.2:
            mods.append("pose")
        if self.gait_embedding and self.quality.get("gait", 0.0) > 0.2:
            mods.append("gait")
        if self.height_estimate > 100.0:
            mods.append("height")
        if self.object_attributes:
            mods.append("carried_objects")
        if len(self.trajectory) >= 2:
            mods.append("trajectory")
        return mods

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "duration_sec": round(max(0.0, self.timestamp_end - self.timestamp_start), 1),
            "face": self.face,
            "partial_face": self.partial_face,
            "body_embedding_len": len(self.body_embedding),
            "body_embedding": self.body_embedding,
            "clothing": self.clothing,
            "pose_sequence_len": len(self.pose_sequence),
            "gait_embedding_len": len(self.gait_embedding),
            "gait_embedding": self.gait_embedding,
            "height_estimate": round(self.height_estimate, 1),
            "object_attributes": self.object_attributes,
            "trajectory_points": len(self.trajectory),
            "trajectory": self.trajectory,
            "quality": {k: round(v, 3) for k, v in self.quality.items()},
            "direction_of_travel": self.direction_of_travel,
            "available_modalities": self.get_available_modalities(),
            "face_available": self.is_face_available(),
            "partial_face_available": self.is_partial_face_available()
        }

    @classmethod
    def from_observation_dict(cls, track_id: str, camera_id: str, obs_dict: Dict[str, Any]) -> "PersonEvidencePacket":
        """Construct packet from track observation data."""
        face_info = obs_dict.get("face", {})
        partial_face_info = obs_dict.get("partial_face", {}) or face_info.get("tier_details", {})
        body_emb = obs_dict.get("body_embedding", [])
        clothing_info = obs_dict.get("clothing", {}) or {
            "clothing_upper": obs_dict.get("clothing_upper", "#334455"),
            "clothing_lower": obs_dict.get("clothing_lower", "#112233")
        }
        gait_emb = obs_dict.get("gait_embedding", [])
        h_est = float(obs_dict.get("estimated_height_cm", obs_dict.get("height_cm", 170.0)))
        objs = obs_dict.get("object_attributes", []) or obs_dict.get("carried_objects", [])

        # Infer quality scores
        f_qual = float(face_info.get("quality_score", 0.85 if face_info.get("face_visible") else 0.0))
        b_qual = 0.85 if body_emb else 0.0
        g_qual = 0.80 if gait_emb else 0.40
        p_qual = float(obs_dict.get("posture_score", 0.85))
        h_qual = 0.85 if h_est > 120 else 0.0
        c_qual = 0.85 if clothing_info else 0.0

        return cls(
            track_id=track_id,
            camera_id=camera_id,
            timestamp_start=float(obs_dict.get("timestamp_start", time.time())),
            timestamp_end=float(obs_dict.get("timestamp_end", time.time())),
            face=face_info,
            partial_face=partial_face_info,
            body_embedding=body_emb,
            clothing=clothing_info,
            pose_sequence=obs_dict.get("pose_sequence", []),
            gait_embedding=gait_emb,
            height_estimate=h_est,
            object_attributes=objs,
            trajectory=obs_dict.get("trajectory", []),
            quality={
                "face": f_qual,
                "body": b_qual,
                "gait": g_qual,
                "pose": p_qual,
                "height": h_qual,
                "clothing": c_qual,
                "trajectory": 0.80 if obs_dict.get("trajectory") else 0.0
            },
            direction_of_travel=obs_dict.get("direction", "UNKNOWN")
        )
