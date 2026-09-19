"""Multi-Modal Evidence Fusion Engine dynamically combining Face, Body, Gait, and Height biometrics."""

import math
from typing import Dict, Any, Tuple
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
    # Max possible distance is sqrt(3 * 255^2) ~= 441.67
    sim = max(0.0, 1.0 - (dist / 220.0))
    return round(sim, 3)

class EvidenceFusionEngine:
    def __init__(self):
        pass

    def evaluate_candidate(self, track: TrackObservation, suspect: CriminalRecord) -> Dict[str, Any]:
        """Perform multi-criteria evidence fusion between a live CCTV track and a known criminal record."""
        # 1. FACE EVIDENCE
        face_available = track.face_visible and track.face_status != "UNAVAILABLE"
        if face_available:
            mask_track = (track.face_status == "MASKED_LOWER")
            face_score = compute_partial_face_similarity(track.face_embedding, suspect.face_embedding, mask_track, False)
            face_evidence_status = "PARTIAL_UPPER_FACE" if mask_track else "FULL_FACE_AVAILABLE"
        else:
            face_score = 0.0
            face_evidence_status = "UNAVAILABLE (Masked/Rear/Blur)"

        # 2. BODY BIOMETRICS & PROPORTIONS
        body_emb_sim = cosine_similarity(track.body_embedding, suspect.body_embedding)

        # Torso / Leg Ratio similarity
        track_ratio = track.body_proportions.get("torso_leg_ratio", 0.85)
        ratio_diff = abs(track_ratio - suspect.torso_leg_ratio)
        ratio_score = max(0.0, 1.0 - (ratio_diff / 0.22))

        # Clothing match
        upper_sim = compute_color_similarity(track.clothing_upper, suspect.clothing_upper_color)
        lower_sim = compute_color_similarity(track.clothing_lower, suspect.clothing_lower_color)
        clothing_score = 0.6 * upper_sim + 0.4 * lower_sim

        body_score = round(0.45 * body_emb_sim + 0.30 * ratio_score + 0.25 * clothing_score, 3)

        # 3. GAIT & POSTURE DYNAMICS
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
        # 0 diff -> 1.0, 10cm diff -> 0.4, >15cm -> 0.0
        height_score = round(max(0.0, 1.0 - (h_diff / 14.0)), 3)

        # 5. DYNAMIC WEIGHT FUSION
        if face_available:
            w_face = 0.40
            w_body = 0.25
            w_gait = 0.25
            w_height = 0.10
        else:
            # Shift weight dynamically to body, gait, and height!
            w_face = 0.00
            w_body = 0.40
            w_gait = 0.45
            w_height = 0.15

        total_confidence = round(
            (w_face * face_score) +
            (w_body * body_score) +
            (w_gait * gait_score) +
            (w_height * height_score),
            3
        )

        # 6. INVESTIGATION STATUS DETERMINATION
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
            "status": status,
            "recommendation": recommendation,
            "is_face_available": face_available,
            "face_evidence_status": face_evidence_status,
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
                "suspect_posture_score": suspect.posture_correctness
            }
        }
