"""Qualitative Confidence Grading & Evidence Narrative Synthesis.

Avoids opaque single match percentages (e.g. 97.3%) and instead generates defensible,
transparent legal and investigative evidence summaries.
"""

from typing import Dict, Any, List

def format_evidence_narrative(evidence_breakdown: Dict[str, Any]) -> str:
    """Generate human-readable narrative explaining why two tracks are associated."""
    lines = []

    face = evidence_breakdown.get("face", {})
    if face.get("available") and face.get("grade") in ["STRONG", "MODERATE"]:
        lines.append(f"Face: {face.get('summary', 'available')}")
    else:
        lines.append("Face: unavailable / unobservable")

    body = evidence_breakdown.get("body", {})
    if body.get("available"):
        lines.append(f"Body: {body.get('grade', 'SUPPORTING').lower()} evidence ({body.get('summary', '')})")

    gait = evidence_breakdown.get("gait", {})
    if gait.get("available"):
        lines.append(f"Gait: {gait.get('grade', 'SUPPORTING').lower()} evidence ({gait.get('summary', '')})")

    clothing = evidence_breakdown.get("clothing", {})
    if clothing.get("available"):
        lines.append(f"Clothing: {clothing.get('grade', 'SUPPORTING').lower()} evidence")

    carried = evidence_breakdown.get("carried_objects", {})
    if carried.get("available") and carried.get("grade") == "STRONG":
        lines.append(f"Accessory: {carried.get('summary', 'detected')}")

    trajectory = evidence_breakdown.get("trajectory", {})
    if trajectory.get("consistent", True):
        lines.append(f"Trajectory: consistent ({trajectory.get('direction', 'Northward')})")
    else:
        lines.append("Trajectory: inconsistent")

    travel = evidence_breakdown.get("travel_time", {})
    if travel.get("is_feasible", True):
        lines.append(f"Travel time: consistent ({travel.get('summary', 'realistic transit pace')})")
    else:
        lines.append(f"Travel time: inconsistent ({travel.get('summary', 'velocity violation')})")

    return " | ".join(lines)
