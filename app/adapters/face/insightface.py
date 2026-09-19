"""InsightFace / ArcFace Adapter with 3-Tier Partial Face CCTV Decomposition.

Extracts facial features, quality scores, and 3-tier spatial partitions
(forehead/eyes, nose/cheeks, mouth/chin) to maintain identification capability
even when suspects wear masks, helmets, or turn partially away.

NOTE ON LICENSING:
The InsightFace code is MIT licensed, but default pretrained models (e.g. buffalo_l, antelopev2)
have non-commercial research licensing terms. Commercial deployments must use commercial-licensed
weights or in-house retrained models.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from app.adapters.base import FaceAnalysisResult
from app.adapters.face.base import BaseCCTVFaceAnalyzer
from app.features.face import FaceAnalyzer


class InsightFaceArcFaceAdapter(BaseCCTVFaceAnalyzer):
    """InsightFace / ArcFace / YuNet-SFace pipeline adapter with CCTV partial face and quality handling."""

    def __init__(
        self,
        det_size: Tuple[int, int] = (640, 640),
        quality_thresh: float = 0.40,
        model_name: str = "buffalo_l"
    ):
        self.det_size = det_size
        self.quality_thresh = quality_thresh
        self.model_name = model_name
        self.backend = "INSIGHTFACE_3TIER_PARTITION"
        self.insight_app = None
        self.yunet_path = None
        self.sface_model = None
        self.fallback_analyzer = FaceAnalyzer()

        try:
            import insightface  # type: ignore
            self.insight_app = insightface.app.FaceAnalysis(name=model_name)
            self.insight_app.prepare(ctx_id=-1, det_size=det_size)
            self.backend = f"INSIGHTFACE_{model_name.upper()}"
        except Exception:
            # Check for local OpenCV Zoo YuNet + SFace neural models in models/
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            yp = os.path.join(base_dir, "models", "face_detection_yunet_2023mar.onnx")
            sp = os.path.join(base_dir, "models", "face_recognition_sface_2021dec.onnx")
            if os.path.isfile(yp) and os.path.isfile(sp):
                self.yunet_path = yp
                try:
                    self.sface_model = cv2.FaceRecognizerSF.create(sp, "")
                    self.backend = "OPENCV_YUNET_SFACE_NEURAL_PIPELINE"
                except Exception:
                    self.backend = "CCTV_QUALITY_AWARE_PARTIAL_FACE_ANALYZER"
            else:
                self.backend = "CCTV_QUALITY_AWARE_PARTIAL_FACE_ANALYZER"

    def analyze_face(self, person_crop: np.ndarray) -> FaceAnalysisResult:
        """Extract quality score, 3-tier decomposition, and normalized face embeddings."""
        if person_crop is None or person_crop.size == 0:
            return FaceAnalysisResult(
                bbox=(0, 0, 0, 0),
                quality_score=0.0,
                is_available=False,
                status="UNAVAILABLE"
            )

        ph, pw = person_crop.shape[:2]
        face_dict = self.fallback_analyzer.analyze_person_crop(person_crop)
        is_visible = face_dict.get("face_visible", False)
        status = face_dict.get("face_status", "UNAVAILABLE")
        fbox = face_dict.get("face_box", [0, 0, 0, 0])
        bw = max(0, fbox[2] - fbox[0])
        bh = max(0, fbox[3] - fbox[1])
        quality = 0.85 if is_visible else 0.10

        if not is_visible:
            return FaceAnalysisResult(
                bbox=(fbox[0], fbox[1], bw, bh),
                quality_score=quality,
                is_available=False,
                status=status,
                tier_embeddings=face_dict.get("tiers", {}),
                full_embedding=face_dict.get("face_embedding", [])
            )

        # If InsightFace is initialized, enrich with ArcFace 512-d embedding
        if self.insight_app is not None:
            try:
                faces = self.insight_app.get(person_crop)
                if faces and len(faces) > 0:
                    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    arcface_emb = best_face.embedding.tolist()
                    norm = np.linalg.norm(arcface_emb)
                    if norm > 0:
                        arcface_emb = [round(float(x / norm), 6) for x in arcface_emb]

                    return FaceAnalysisResult(
                        bbox=(
                            int(best_face.bbox[0]), int(best_face.bbox[1]),
                            int(best_face.bbox[2] - best_face.bbox[0]), int(best_face.bbox[3] - best_face.bbox[1])
                        ),
                        quality_score=float(best_face.det_score),
                        is_available=True,
                        status=status,
                        tier_embeddings=face_dict.get("tiers", {}),
                        full_embedding=arcface_emb
                    )
            except Exception:
                pass

        # If OpenCV Zoo YuNet neural face detector is initialized
        if self.yunet_path is not None and pw >= 20 and ph >= 40:
            try:
                yn_det = cv2.FaceDetectorYN.create(self.yunet_path, "", (pw, ph), score_threshold=self.quality_thresh)
                _, yfaces = yn_det.detect(person_crop)
                if yfaces is not None and len(yfaces) > 0:
                    best = yfaces[0]
                    fx, fy, fw, fh = int(best[0]), int(best[1]), int(best[2]), int(best[3])
                    fx = max(0, min(fx, pw - 1))
                    fy = max(0, min(fy, ph - 1))
                    fw = max(1, min(fw, pw - fx))
                    fh = max(1, min(fh, ph - fy))
                    score = float(best[-1])

                    emb_sface = []
                    if self.sface_model is not None:
                        try:
                            aligned = self.sface_model.alignCrop(person_crop, best)
                            raw_feat = self.sface_model.feature(aligned)
                            norm = np.linalg.norm(raw_feat)
                            if norm > 0:
                                raw_feat = raw_feat / norm
                            emb_sface = [round(float(x), 6) for x in raw_feat.flatten()]
                        except Exception:
                            pass

                    return FaceAnalysisResult(
                        bbox=(fx, fy, fw, fh),
                        quality_score=round(score, 3),
                        is_available=True,
                        status="FULL_FACE" if score >= 0.60 else "PARTIAL_UPPER",
                        tier_embeddings=face_dict.get("tiers", {}),
                        full_embedding=emb_sface if emb_sface else face_dict.get("face_embedding", [])
                    )
            except Exception:
                pass

        return FaceAnalysisResult(
            bbox=(fbox[0], fbox[1], bw, bh),
            quality_score=quality,
            is_available=True,
            status=status,
            tier_embeddings=face_dict.get("tiers", {}),
            full_embedding=face_dict.get("face_embedding", [])
        )

    def get_backend_info(self) -> Dict[str, Any]:
        is_neural = (self.insight_app is not None) or (self.yunet_path is not None)
        return {
            "model_id": "insightface",
            "name": "InsightFace / OpenCV Zoo (YuNet + SFace)",
            "backend": self.backend,
            "model_name": self.model_name,
            "quality_threshold": self.quality_thresh,
            "decomposition": "3-Tier (Upper 0-33%, Mid 33-66%, Lower 66-100%)",
            "is_neural_model_loaded": is_neural,
            "license_warning": "Pretrained recognition models are Non-Commercial Research Only; in-house commercial weights required for enterprise."
        }
