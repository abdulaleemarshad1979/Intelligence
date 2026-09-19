"""Multi-Modal Evidence Evaluation Suite.

Provides transparent, defensible evidence grading across modalities:
Face, Body, Gait, Pose, Clothing, Height, and Carried Objects.
"""

from app.evidence.face import FaceEvidenceEvaluator
from app.evidence.body import BodyEvidenceEvaluator
from app.evidence.gait import GaitEvidenceEvaluator
from app.evidence.pose import PoseEvidenceEvaluator
from app.evidence.clothing import ClothingEvidenceEvaluator
from app.evidence.height import HeightEvidenceEvaluator
from app.evidence.object_features import CarriedObjectEvidenceEvaluator
from app.evidence.packet import PersonEvidencePacket

__all__ = [
    "PersonEvidencePacket",
    "FaceEvidenceEvaluator",
    "BodyEvidenceEvaluator",
    "GaitEvidenceEvaluator",
    "PoseEvidenceEvaluator",
    "ClothingEvidenceEvaluator",
    "HeightEvidenceEvaluator",
    "CarriedObjectEvidenceEvaluator"
]
