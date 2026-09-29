"""Multi-Modal Evidence Fusion Engine dynamically combining Face, Body, Gait, and Height biometrics
with built-in humility vetoes and calibrated confidence scoring for police-grade accuracy."""

import json
import math
from typing import Dict, Any, Tuple, List
from app.database.models import CriminalRecord, TrackObservation
from app.reid.embedding import cosine_similarity
from app.features.partial_face import compute_partial_face_similarity
from app.fusion.disparity_veto import DisparityVetoGate


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

    Section 3: Signal Fusion Architecture & Disparity Veto:
    "At 1:N scale across an entire city or district gallery, soft biometrics alone yield
    unacceptably high false-match rates. Soft signals must act as conditional confirmations
    or hard geometric pruning gates rather than independent identity verifiers."
    """

    def __init__(self, height_veto_threshold_cm: float = 12.0, enforce_1_to_n_soft_cap: bool = True):
        self.height_veto_threshold_cm = height_veto_threshold_cm
        self.enforce_1_to_n_soft_cap = enforce_1_to_n_soft_cap
        self.disparity_gate = DisparityVetoGate(max_height_disparity_cm=height_veto_threshold_cm)

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

            est_h = track.get("estimated_height_cm")
            stride_h = track.get("stride_length_cm")
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
                estimated_height_cm=float(est_h) if (est_h is not None and float(est_h) > 0) else 0.0,
                body_proportions=parse_if_json(track.get("body_proportions"), {}),
                clothing_upper=str(track.get("clothing_upper", "#000000") or "#000000"),
                clothing_lower=str(track.get("clothing_lower", "#000000") or "#000000"),
                carried_objects=parse_if_json(track.get("carried_objects"), []),
                stride_length_px=float(track.get("stride_length_px", 0.0) or 0.0),
                stride_length_cm=float(stride_h) if (stride_h is not None and float(stride_h) > 0) else 0.0,
                cadence_steps_per_sec=float(track.get("cadence_steps_per_sec", 0.0) or 0.0),
                spine_tilt_deg=float(track.get("spine_tilt_deg", 0.0) or 0.0),
                posture_score=float(track.get("posture_score", 0.0) or 0.0),
                gait_wave=parse_if_json(track.get("gait_wave"), []),
                face_embedding=parse_if_json(track.get("face_embedding"), []),
                body_embedding=parse_if_json(track.get("body_embedding"), []),
                gait_embedding=parse_if_json(track.get("gait_embedding"), [])
            )

        # 1. FACE EVIDENCE (YuNet + SFace / ArcFace)
        face_available = bool(
            track.face_visible and
            track.face_status != "UNAVAILABLE" and
            getattr(track, "face_embedding", None) and
            len(track.face_embedding) > 0 and
            getattr(suspect, "face_embedding", None) and
            len(suspect.face_embedding) > 0
        )
        if face_available:
            mask_track = (track.face_status == "MASKED_LOWER")
            face_score = compute_partial_face_similarity(track.face_embedding, suspect.face_embedding, mask_track, False)
            face_evidence_status = "PARTIAL_UPPER_FACE" if mask_track else "FULL_FACE_AVAILABLE"
        else:
            face_score = 0.0
            face_evidence_status = "UNAVAILABLE (Masked/Rear/Blur)"

        # 2. BODY BIOMETRICS & PROPORTIONS (OSNet Re-ID Backbone)
        body_available = bool(
            getattr(track, "body_embedding", None) and
            len(track.body_embedding) > 0 and
            getattr(suspect, "body_embedding", None) and
            len(suspect.body_embedding) > 0
        )
        if body_available:
            body_emb_sim = cosine_similarity(track.body_embedding, suspect.body_embedding)
            track_ratio = track.body_proportions.get("torso_leg_ratio", 0.85) if getattr(track, "body_proportions", None) else 0.85
            ratio_diff = abs(track_ratio - suspect.torso_leg_ratio)
            ratio_score = max(0.0, 1.0 - (ratio_diff / 0.22))
            upper_sim = compute_color_similarity(track.clothing_upper, suspect.clothing_upper_color)
            lower_sim = compute_color_similarity(track.clothing_lower, suspect.clothing_lower_color)
            clothing_score = 0.6 * upper_sim + 0.4 * lower_sim
            suspect_items = getattr(suspect, "carried_objects", []) or []
            track_items = getattr(track, "carried_objects", []) or []
            if suspect_items:
                common = set(suspect_items).intersection(set(track_items))
                carried_score = len(common) / max(1, len(suspect_items))
                body_score = round(0.40 * body_emb_sim + 0.25 * ratio_score + 0.20 * clothing_score + 0.15 * carried_score, 3)
            else:
                carried_score = 1.0
                body_score = round(0.45 * body_emb_sim + 0.30 * ratio_score + 0.25 * clothing_score, 3)
        else:
            body_emb_sim = 0.0
            body_score = None
            track_ratio = 0.85
            track_items = []
            suspect_items = getattr(suspect, "carried_objects", []) or []

        # 3. GAIT & POSTURE DYNAMICS (Bonus Vote / Nudge Modifier)
        track_stride = getattr(track, "stride_length_cm", 0.0) or 0.0
        suspect_stride = getattr(suspect, "stride_length_cm", 0.0) or 0.0
        gait_available = bool(
            track_stride > 10.0 and
            suspect_stride > 10.0 and
            getattr(track, "gait_embedding", None) and
            len(track.gait_embedding) > 0 and
            getattr(suspect, "gait_embedding", None) and
            len(suspect.gait_embedding) > 0
        )
        if gait_available:
            gait_emb_sim = cosine_similarity(track.gait_embedding, suspect.gait_embedding)
            stride_diff = abs(track_stride - suspect_stride)
            stride_score = max(0.0, 1.0 - (stride_diff / 16.0))
            lean_diff = abs(track.spine_tilt_deg - suspect.posture_lean_angle)
            lean_score = max(0.0, 1.0 - (lean_diff / 5.0))
            posture_sim = 1.0 - min(1.0, abs(track.posture_score - suspect.posture_correctness) / 0.4)
            gait_score = round(0.40 * gait_emb_sim + 0.35 * stride_score + 0.15 * lean_score + 0.10 * posture_sim, 3)
        else:
            gait_emb_sim = 0.0
            gait_score = None

        # 4. CALIBRATED HEIGHT STATURE
        track_height = getattr(track, "estimated_height_cm", 0.0) or 0.0
        suspect_height = getattr(suspect, "known_height_cm", 0.0) or 0.0
        height_available = bool(track_height > 100.0 and suspect_height > 100.0)
        if height_available:
            h_diff = abs(track_height - suspect_height)
            height_score = round(max(0.0, 1.0 - (h_diff / 14.0)), 3)
        else:
            h_diff = None
            height_score = None

        # 5. DYNAMIC WEIGHT FUSION (Only score measured modalities)
        active_modalities = {}
        if face_available:
            active_modalities["face"] = (0.40, face_score)
            active_modalities["body"] = (0.25, body_score) if body_available else (0.0, None)
            active_modalities["gait"] = (0.25, gait_score) if gait_available else (0.0, None)
            active_modalities["height"] = (0.10, height_score) if height_available else (0.0, None)
        else:
            active_modalities["face"] = (0.00, 0.0)
            active_modalities["body"] = (0.40, body_score) if body_available else (0.0, None)
            active_modalities["gait"] = (0.45, gait_score) if gait_available else (0.0, None)
            active_modalities["height"] = (0.15, height_score) if height_available else (0.0, None)

        valid_weights = {k: v[0] for k, v in active_modalities.items() if v[1] is not None and v[0] > 0}
        total_w = sum(valid_weights.values())

        if total_w > 0:
            raw_confidence = round(sum(valid_weights[k] * active_modalities[k][1] for k in valid_weights) / total_w, 3)
            weights_used = {
                "face": round(valid_weights.get("face", 0.0) / total_w, 2),
                "body": round(valid_weights.get("body", 0.0) / total_w, 2),
                "gait": round(valid_weights.get("gait", 0.0) / total_w, 2),
                "height": round(valid_weights.get("height", 0.0) / total_w, 2)
            }
        else:
            raw_confidence = 0.0
            weights_used = {"face": 0.0, "body": 0.0, "gait": 0.0, "height": 0.0}

        # 6. BIOMETRIC HUMILITY & CONTRADICTION VETOES
        veto_reasons: List[str] = []
        is_vetoed = False
        is_pruned = False

        # Hard Geometric Pruning Gate (Section 3: Disparity Veto)
        if height_available and gait_available:
            is_pruned, pruning_reasons = self.disparity_gate.evaluate_geometric_pruning(
                track_height_cm=track_height,
                suspect_height_cm=suspect_height,
                track_ratio=track_ratio,
                suspect_ratio=suspect.torso_leg_ratio,
                track_stride_cm=track_stride,
                suspect_stride_cm=suspect_stride
            )
            if is_pruned:
                is_vetoed = True
                veto_reasons.extend(pruning_reasons)

        # Height contradiction: cannot grow or shrink by >12cm (Humility Veto)
        if height_available and h_diff is not None and h_diff > self.height_veto_threshold_cm:
            if not any("Height disparity" in r for r in veto_reasons):
                is_vetoed = True
                veto_reasons.append(
                    f"HUMILITY_VETO: Height disparity ({h_diff:.1f}cm > {self.height_veto_threshold_cm:.1f}cm) "
                    f"contradicts suspect profile (Track: {track_height:.0f}cm vs Suspect: {suspect_height:.0f}cm)."
                )

        # Facial contradiction: if face is fully visible but SFace score is very poor, reject match
        if face_available and face_score < 0.25:
            is_vetoed = True
            veto_reasons.append(
                f"HUMILITY_VETO: Frontal face visible but facial similarity ({face_score:.2f}) contradicts suspect face."
            )

        # Body contradiction: when face is unavailable and body embedding similarity is low, gait alone cannot trigger match
        if not face_available and body_available and body_emb_sim < 0.22:
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
            if face_available:
                total_confidence = raw_confidence
                if total_confidence >= 0.78:
                    status = "HIGH_CONFIDENCE"
                    recommendation = "Immediate Authorized Intercept & Verification"
                elif total_confidence >= 0.52:
                    status = "REVIEW_REQUIRED"
                    recommendation = "Candidate Resembles Reference: Human Investigator Review Required"
                else:
                    status = "UNKNOWN_PERSON"
                    recommendation = "Non-Matching Track / Incident Logging Only"
            else:
                if self.enforce_1_to_n_soft_cap:
                    # Section 3: Soft biometrics cannot independently verify identity at 1:N scale
                    total_confidence = min(raw_confidence, 0.45)
                    if total_confidence >= 0.40:
                        status = "REVIEW_REQUIRED"
                        recommendation = "Conditional Confirmation Only: Mandatory Human Review (Soft biometrics cannot verify identity at 1:N scale)"
                    else:
                        status = "UNKNOWN_PERSON"
                        recommendation = "Non-Matching Track / Incident Logging Only"
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
            "is_pruned": is_pruned,
            "disparity_veto_triggered": is_vetoed,
            "signal_fusion_rule": "At 1:N scale across city gallery, soft biometrics act as conditional confirmations or hard geometric pruning gates rather than independent identity verifiers.",
            "veto_reasons": veto_reasons,
            "humility_status": "VETO_TRIGGERED" if is_vetoed else "PASSED",
            "scores": {
                "face_score": round(face_score, 3) if face_available else 0.0,
                "body_score": body_score,
                "gait_score": gait_score,
                "height_score": height_score
            },
            "weights_used": weights_used,
            "biometric_comparison": {
                "estimated_height_cm": track_height if height_available else None,
                "known_height_cm": suspect_height if suspect_height > 0 else None,
                "height_delta_cm": round(h_diff, 1) if h_diff is not None else None,
                "track_stride_cm": track_stride if gait_available else None,
                "suspect_stride_cm": suspect_stride if suspect_stride > 0 else None,
                "track_torso_leg_ratio": track_ratio if body_available else None,
                "suspect_torso_leg_ratio": suspect.torso_leg_ratio,
                "track_clothing": {"upper": track.clothing_upper, "lower": track.clothing_lower} if body_available else None,
                "suspect_clothing": {"upper": suspect.clothing_upper_color, "lower": suspect.clothing_lower_color},
                "track_posture_score": track.posture_score if gait_available else None,
                "suspect_posture_score": suspect.posture_correctness,
                "track_carried_objects": track_items if body_available else [],
                "suspect_carried_objects": suspect_items,
                "carried_objects_matched": bool(set(suspect_items).intersection(set(track_items))) if (body_available and suspect_items) else True
            }
        }
