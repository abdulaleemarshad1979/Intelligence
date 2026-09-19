"""Face analysis adapters package."""

from app.adapters.face.base import BaseCCTVFaceAnalyzer
from app.adapters.face.insightface import InsightFaceArcFaceAdapter

__all__ = ["BaseCCTVFaceAnalyzer", "InsightFaceArcFaceAdapter"]
