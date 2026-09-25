"""Signal Fusion Architecture & Disparity Veto Engine.

Core Principle:
"At 1:N scale across an entire city or district gallery, soft biometrics alone yield
unacceptably high false-match rates. Soft signals must act as conditional confirmations
or hard geometric pruning gates rather than independent identity verifiers."

Law enforcement systems operating across large municipal galleries cannot allow soft
signals (height, clothing, body proportions, gait kinematics) to trigger automated arrests
or high-confidence alerts in isolation. Soft biometrics are strictly:
1. Hard Geometric Pruning Gates: Rejecting physically impossible matches before 1:N ranking.
2. Conditional Confirmations: Boosting or validating primary hard biometrics (face/FRS),
   or proposing candidates strictly for mandatory human investigator review.
"""

from typing import Dict, Any, List, Tuple, Optional


class DisparityVetoGate:
    """Hard geometric pruning gate and conditional confirmation controller for 1:N gallery retrieval."""

    def __init__(
        self,
        max_height_disparity_cm: float = 12.0,
        max_ratio_disparity: float = 0.30,
        max_stride_disparity_cm: float = 25.0,
        max_soft_only_confidence: float = 0.45,
        min_primary_face_threshold: float = 0.30
    ):
        self.max_height_disparity_cm = max_height_disparity_cm
        self.max_ratio_disparity = max_ratio_disparity
        self.max_stride_disparity_cm = max_stride_disparity_cm
        self.max_soft_only_confidence = max_soft_only_confidence
        self.min_primary_face_threshold = min_primary_face_threshold

    def evaluate_geometric_pruning(
        self,
        track_height_cm: float,
        suspect_height_cm: float,
        track_ratio: float,
        suspect_ratio: float,
        track_stride_cm: float,
        suspect_stride_cm: float
    ) -> Tuple[bool, List[str]]:
        """Evaluate hard geometric invariant bounds to prune candidates before 1:N ranking."""
        pruning_reasons: List[str] = []
        is_pruned = False

        # 1. Height Disparity Gate (Height cannot physically change by > 12cm)
        height_delta = abs(track_height_cm - suspect_height_cm)
        if height_delta > self.max_height_disparity_cm:
            is_pruned = True
            pruning_reasons.append(
                f"DISPARITY_VETO: Height disparity ({height_delta:.1f}cm > {self.max_height_disparity_cm:.1f}cm) "
                f"violates geometric invariant (Live: {track_height_cm:.0f}cm vs Gallery: {suspect_height_cm:.0f}cm)."
            )

        # 2. Torso-to-Leg Ratio Gate
        ratio_delta = abs(track_ratio - suspect_ratio)
        if ratio_delta > self.max_ratio_disparity:
            is_pruned = True
            pruning_reasons.append(
                f"DISPARITY_VETO: Skeletal proportion disparity (Ratio Δ {ratio_delta:.2f} > {self.max_ratio_disparity:.2f}) "
                f"violates anatomical proportions."
            )

        # 3. Kinematic Stride Disparity Gate
        if track_stride_cm > 10.0 and suspect_stride_cm > 10.0:
            stride_delta = abs(track_stride_cm - suspect_stride_cm)
            if stride_delta > self.max_stride_disparity_cm:
                is_pruned = True
                pruning_reasons.append(
                    f"DISPARITY_VETO: Stride kinematics disparity ({stride_delta:.1f}cm > {self.max_stride_disparity_cm:.1f}cm) "
                    f"incompatible with subject gait profile."
                )

        return is_pruned, pruning_reasons

    def evaluate_fusion_policy(
        self,
        raw_confidence: float,
        is_face_available: bool,
        face_score: float,
        is_pruned: bool,
        pruning_reasons: List[str],
        scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Apply Section 3 Signal Fusion policy enforcing conditional confirmations."""
        # Case A: Hard Geometric Pruning Gate Triggered
        if is_pruned:
            return {
                "is_vetoed": True,
                "status": "UNKNOWN_PERSON",
                "pruned_by_geometry": True,
                "confidence": min(raw_confidence, 0.38),
                "policy_decision": "PRUNED_GEOMETRIC_DISPARITY",
                "recommendation": f"Hard Geometric Pruning Gate: Suppressed ({'; '.join(pruning_reasons)})",
                "explanation": "At 1:N scale, candidate violates physical geometry bounds and was pruned."
            }

        # Case B: Primary Hard Biometric Available (Frontal or Partial Face)
        if is_face_available and face_score >= self.min_primary_face_threshold:
            # Soft signals act as conditional confirmations boosting primary score
            if raw_confidence >= 0.78:
                status = "HIGH_CONFIDENCE"
                rec = "Immediate Authorized Intercept & Verification"
            elif raw_confidence >= 0.52:
                status = "REVIEW_REQUIRED"
                rec = "Multi-Criteria Candidate Match: Human Investigator Review Required"
            else:
                status = "UNKNOWN_PERSON"
                rec = "Non-Matching Track / Incident Logging Only"

            return {
                "is_vetoed": False,
                "status": status,
                "pruned_by_geometry": False,
                "confidence": raw_confidence,
                "policy_decision": "CONFIRMED_PRIMARY_WITH_SOFT_BIOMETRICS",
                "recommendation": rec,
                "explanation": "Primary face biometric confirmed; soft biometrics act as conditional confirmations."
            }

        # Case C: Soft Biometrics Only (Face Masked / Occluded / Rear View)
        # CRITICAL RULE: Soft biometrics alone yield unacceptably high false-match rates at 1:N scale.
        # They CANNOT act as independent identity verifiers or produce automatic HIGH_CONFIDENCE matches.
        soft_cap = self.max_soft_only_confidence
        calibrated_conf = min(raw_confidence, soft_cap)

        if raw_confidence >= 0.50:
            status = "REVIEW_REQUIRED"
            rec = "Conditional Confirmation Only: Mandatory Human Review (Soft biometrics cannot verify identity at 1:N scale)"
        else:
            status = "UNKNOWN_PERSON"
            rec = "Non-Matching Track / Incident Logging Only"

        return {
            "is_vetoed": False,
            "status": status,
            "pruned_by_geometry": False,
            "confidence": calibrated_conf,
            "policy_decision": "CONDITIONAL_CONFIRMATION_ONLY",
            "recommendation": rec,
            "explanation": "At 1:N scale across city gallery, soft biometrics act strictly as conditional confirmations."
        }
