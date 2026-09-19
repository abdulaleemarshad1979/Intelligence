"""Multi-Modal Evidence Fusion Engine dynamically combining Face, Body, Gait, and Height biometrics
with built-in humility vetoes and calibrated confidence scoring for police-grade accuracy."""

import json
import math
from typing import Dict, Any, Tuple, List
from app.database.models import CriminalRecord, TrackObservation
from app.reid.embedding import cosine_similarity
from app.features.partial_face import compute_partial_face_similarity


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    hex_clean = hex_str.lstrip('#')
    if len(hex_clean) != 6:
        return (50, 50, 50)
    return (int(hex_clean[0:2], 16), int(hex_clean[2:4], 16), int(hex_clean[4:6], 16))


def compute_color_similarity(c1_hex: str, c2_hex: str) -> float:
    r1, g1, b1 = hex_to_rgb(c1_hex)
    r2, g2, b2 = hex_to_rgb(c2_hex)
    dist = math.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)
    sim = max(0.0, 1.0 - (dist / 220.0))
    return round(sim, 3)


class EvidenceFusionEngine:
    """Combines facial, body Re-ID, skeletal posture, gait bonus, and calibrated stature.

    Integrates 'Humility Vetoes':
    A system that says 'I don't know' is essential for law enforcement credibility.
    If physical attributes contradict (e.g. height difference > 12cm), false matches are
    actively suppressed and flagged for human investigator review.
    """

    def __init__(self, height_veto_threshold_cm: float = 12.0):
        self.height_veto_threshold_cm = height_veto_threshold_cm

    def evaluate_candidate(self, track: Any, suspect: CriminalRecord) -> Dict[str, Any]:
        """Perform multi-criteria evidence fusion between a live CCTV track and a known record."""
        if isinstance(track, dict):
            def parse_if_json(val, default):
                if isinstance(val, str):
                    try:
                        return json.loads(val)
                    except Exception:
                        return default
                return val if val is not None else default

            track = TrackObservation(
                track_id=str(track.get("track_id", "0001")),
                camera_id=str(track.get("camera_id", "CAM-001")),
                first_seen=float(track.get("first_seen", 0.0) or 0.0),
                last_seen=float(track.get("last_seen", 0.0) or 0.0),
                frame_count=int(track.get("frame_count", 0) or 0),
                best_frame_path=str(track.get("best_frame_path", "")),
                face_visible=bool(track.get("face_visible", False)),
                face_status=str(track.get("face_status", "UNAVAILABLE")),
                face_tier_details=parse_if_json(track.get("face_tier_details"), {}),
                estimated_height_cm=float(track.get("estimated_height_cm", 170.0) or 170.0),
                body_proportions=parse_if_json(track.get("body_proportions"), {}),
                clothing_upper=str(track.get("clothing_upper", "#000000") or "#000000"),
                clothing_lower=str(track.get("clothing_lower", "#000000") or "#000000"),
                carried_objects=parse_if_json(track.get("carried_objects"), []),
                stride_length_px=float(track.get("stride_length_px", 0.0) or 0.0),
                stride_length_cm=float(track.get("stride_length_cm", 65.0) or 65.0),
                cadence_steps_per_sec=float(track.get("cadence_steps_per_sec", 0.0) or 0.0),
                spine_tilt_deg=float(track.get("spine_tilt_deg", 0.0) or 0.0),
                posture_score=float(track.get("posture_score", 0.85) or 0.85),
                gait_wave=parse_if_json(track.get("gait_wave"), []),
                face_embedding=parse_if_json(track.get("face_embedding"), []),
                body_embedding=parse_if_json(track.get("body_embedding"), []),
                gait_embedding=parse_if_json(track.get("gait_embedding"), [])
            )

        # 1. FACE EVIDENCE (YuNet + SFace / ArcFace)
        face_available = track.face_visible and track.face_status != "UNAVAILABLE"
        if face_available:
            mask_track = (track.face_status == "MASKED_LOWER")
            face_score = compute_partial_face_similarity(track.face_embedding, suspect.face_embedding, mask_track, False)
            face_evidence_status = "PARTIAL_UPPER_FACE" if mask_track else "FULL_FACE_AVAILABLE"
        else:
            face_score = 0.0
            face_evidence_status = "UNAVAILABLE (Masked/Rear/Blur)"

        # 2. BODY BIOMETRICS & PROPORTIONS (OSNet Re-ID Backbone)
        body_emb_sim = cosine_similarity(track.body_embedding, suspect.body_embedding)

        # Torso / Leg Ratio similarity
        track_ratio = track.body_proportions.get("torso_leg_ratio", 0.85)
        ratio_diff = abs(track_ratio - suspect.torso_leg_ratio)
        ratio_score = max(0.0, 1.0 - (ratio_diff / 0.22))

        # Clothing match
        upper_sim = compute_color_similarity(track.clothing_upper, suspect.clothing_upper_color)
        lower_sim = compute_color_similarity(track.clothing_lower, suspect.clothing_lower_color)
        clothing_score = 0.6 * upper_sim + 0.4 * lower_sim

        # Carried objects match (backpacks, bags, etc.)
        suspect_items = getattr(suspect, "carried_objects", []) or []
        track_items = getattr(track, "carried_objects", []) or []
        if suspect_items:
            common = set(suspect_items).intersection(set(track_items))
            carried_score = len(common) / max(1, len(suspect_items))
            body_score = round(0.40 * body_emb_sim + 0.25 * ratio_score + 0.20 * clothing_score + 0.15 * carried_score, 3)
        else:
            carried_score = 1.0
            body_score = round(0.45 * body_emb_sim + 0.30 * ratio_score + 0.25 * clothing_score, 3)

        # 3. GAIT & POSTURE DYNAMICS (Bonus Vote / Nudge Modifier)
        gait_emb_sim = cosine_similarity(track.gait_embedding, suspect.gait_embedding)

        # Stride length match (cm)
        stride_diff = abs(track.stride_length_cm - suspect.stride_length_cm)
        stride_score = max(0.0, 1.0 - (stride_diff / 16.0))

        # Spine lean / posture tilt match
        lean_diff = abs(track.spine_tilt_deg - suspect.posture_lean_angle)
        lean_score = max(0.0, 1.0 - (lean_diff / 5.0))

        # Posture correctness match
        posture_sim = 1.0 - min(1.0, abs(track.posture_score - suspect.posture_correctness) / 0.4)

        gait_score = round(0.40 * gait_emb_sim + 0.35 * stride_score + 0.15 * lean_score + 0.10 * posture_sim, 3)

        # 4. CALIBRATED HEIGHT STATURE
        h_diff = abs(track.estimated_height_cm - suspect.known_height_cm)
        height_score = round(max(0.0, 1.0 - (h_diff / 14.0)), 3)

        # 5. DYNAMIC WEIGHT FUSION
        if face_available:
            w_face = 0.40
            w_body = 0.25
            w_gait = 0.25
            w_height = 0.10
        else:
            w_face = 0.00
            w_body = 0.40
            w_gait = 0.45
            w_height = 0.15

        raw_confidence = round(
            (w_face * face_score) +
            (w_body * body_score) +
            (w_gait * gait_score) +
            (w_height * height_score),
            3
        )

        # 6. BIOMETRIC HUMILITY & CONTRADICTION VETOES
        veto_reasons: List[str] = []
        is_vetoed = False

        # Height contradiction: cannot grow or shrink by >12cm
        if h_diff > self.height_veto_threshold_cm:
            is_vetoed = True
            veto_reasons.append(
                f"HUMILITY_VETO: Height disparity ({h_diff:.1f}cm > {self.height_veto_threshold_cm:.1f}cm) "
                f"contradicts suspect profile (Track: {track.estimated_height_cm:.0f}cm vs Suspect: {suspect.known_height_cm:.0f}cm)."
            )

        # Facial contradiction: if face is fully visible but SFace score is very poor, reject match
        if face_available and face_score < 0.25:
            is_vetoed = True
            veto_reasons.append(
                f"HUMILITY_VETO: Frontal face visible but facial similarity ({face_score:.2f}) contradicts suspect face."
            )

        # Body contradiction: when face is unavailable and body embedding similarity is low, gait alone cannot trigger match
        if not face_available and body_emb_sim < 0.22:
            is_vetoed = True
            veto_reasons.append(
                f"HUMILITY_VETO: Rear/occluded view with poor body embedding similarity ({body_emb_sim:.2f} < 0.22). Match rejected."
            )

        # Final calibrated confidence
        if is_vetoed:
            total_confidence = min(raw_confidence, 0.42)
            status = "UNKNOWN_PERSON"
            recommendation = f"Humility Veto: Potential False Positive Suppressed ({'; '.join(veto_reasons)})"
        else:
            total_confidence = raw_confidence
            if total_confidence >= 0.78:
                status = "HIGH_CONFIDENCE"
                recommendation = "Immediate Authorized Intercept & Verification"
            elif total_confidence >= 0.52:
                status = "REVIEW_REQUIRED"
                recommendation = "Multi-Criteria Candidate Match: Human Investigator Review Required"
            else:
                status = "UNKNOWN_PERSON"
                recommendation = "Non-Matching Track / Incident Logging Only"

        return {
            "suspect_id": suspect.id,
            "suspect_name": suspect.accused_name,
            "fir_no": suspect.fir_no,
            "police_station": suspect.police_station,
            "acts_sec": suspect.acts_sec,
            "total_confidence": total_confidence,
            "raw_confidence": raw_confidence,
            "status": status,
            "recommendation": recommendation,
            "is_face_available": face_available,
            "face_evidence_status": face_evidence_status,
            "is_vetoed": is_vetoed,
            "veto_reasons": veto_reasons,
            "humility_status": "VETO_TRIGGERED" if is_vetoed else "PASSED",
            "scores": {
                "face_score": round(face_score, 3),
                "body_score": body_score,
                "gait_score": gait_score,
                "height_score": height_score
            },
            "weights_used": {
                "face": w_face,
                "body": w_body,
                "gait": w_gait,
                "height": w_height
            },
            "biometric_comparison": {
                "estimated_height_cm": track.estimated_height_cm,
                "known_height_cm": suspect.known_height_cm,
                "height_delta_cm": round(h_diff, 1),
                "track_stride_cm": track.stride_length_cm,
                "suspect_stride_cm": suspect.stride_length_cm,
                "track_torso_leg_ratio": track_ratio,
                "suspect_torso_leg_ratio": suspect.torso_leg_ratio,
                "track_clothing": {"upper": track.clothing_upper, "lower": track.clothing_lower},
                "suspect_clothing": {"upper": suspect.clothing_upper_color, "lower": suspect.clothing_lower_color},
                "track_posture_score": track.posture_score,
                "suspect_posture_score": suspect.posture_correctness,
                "track_carried_objects": track_items,
                "suspect_carried_objects": suspect_items,
                "carried_objects_matched": bool(set(suspect_items).intersection(set(track_items))) if suspect_items else True
            }
        }
