"""Multi-Modal Evidence: Carried-Object Observations & Accessories.

Detects and correlates carried items (backpack, shoulder bag, umbrella, helmet, phone),
providing valuable re-identification persistence across cameras.
"""

from typing import Dict, Any, List, Optional

class CarriedObjectEvidenceEvaluator:
    """Evaluates carried accessories and object consistency."""

    def evaluate(self, probe_feat: Dict[str, Any], cand_feat: Dict[str, Any]) -> Dict[str, Any]:
        probe_objs = [str(x).lower() for x in probe_feat.get("carried_objects", [])]
        cand_objs = [str(x).lower() for x in cand_feat.get("carried_objects", [])]

        if not probe_objs and not cand_objs:
            return {
                "modality": "CARRIED_OBJECTS",
                "available": False,
                "score": 0.5,
                "grade": "NEUTRAL",
                "summary": "No carried accessories detected on either track",
                "details": {
                    "probe_objects": [],
                    "candidate_objects": []
                }
            }

        # Intersect detected objects
        common = set(probe_objs).intersection(set(cand_objs))
        if common:
            matched_str = ", ".join(sorted(list(common)))
            return {
                "modality": "CARRIED_OBJECTS",
                "available": True,
                "score": 1.0,
                "grade": "STRONG",
                "summary": f"Carried accessory confirmed ({matched_str} detected)",
                "details": {
                    "matched_objects": list(common),
                    "probe_objects": probe_objs,
                    "candidate_objects": cand_objs
                }
            }

        if probe_objs and not cand_objs:
            return {
                "modality": "CARRIED_OBJECTS",
                "available": True,
                "score": 0.5,
                "grade": "SUPPORTING",
                "summary": f"Probe carries {probe_objs}; candidate object not clearly resolved",
                "details": {
                    "probe_objects": probe_objs,
                    "candidate_objects": cand_objs
                }
            }

        return {
            "modality": "CARRIED_OBJECTS",
            "available": True,
            "score": 0.4,
            "grade": "SUPPORTING",
            "summary": f"Different accessories detected ({cand_objs} vs {probe_objs})",
            "details": {
                "probe_objects": probe_objs,
                "candidate_objects": cand_objs
            }
        }
