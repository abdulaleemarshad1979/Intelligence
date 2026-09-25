"""Pluggable Model Registry for Dynamic Multi-Modal CCTV Intelligence.

Provides a unified central registry for managing, instantiating, and dynamically
switching perception adapters:
- Detection: YOLO, RT-DETR
- Tracking: ByteTrack, BoT-SORT, NVIDIA DeepStream, MMTracking
- Person Re-ID: Torchreid OSNet, FastReID
- Face Analysis: InsightFace / ArcFace
- Pose Estimation: RTMPose, MMPose
- Gait Recognition: OpenGait, GaitSet
"""

from typing import Dict, Any, List, Optional
from app.adapters.detection import (
    BasePedestrianDetector,
    YOLODetectorAdapter,
    RTDETRDetectorAdapter,
    FullBodyPoseDetectorAdapter,
    EnsemblePedestrianDetector
)
from app.adapters.tracking import (
    BaseMOTTracker, ByteTrackAdapter, BoTSORTAdapter,
    DeepStreamTrackerAdapter, MMTrackingAdapter
)
from app.adapters.reid import BasePersonReIDModel, OSNetReIDAdapter, FastReIDAdapter
from app.adapters.face import BaseCCTVFaceAnalyzer, InsightFaceArcFaceAdapter
from app.adapters.pose import BaseSkeletalPoseEstimator, RTMPoseAdapter, MMPoseAdapter
from app.adapters.gait import GaitAdapter, OpenGaitAdapter, GaitSetAdapter
from app.adapters.gait.handcrafted_kinematics import HandcraftedKinematicsAdapter
from app.adapters.license_audit import LicenseAuditor, MODEL_LICENSE_CATALOG


class ModelRegistry:
    """Singleton registry coordinating active perception models and dynamic switching."""

    def __init__(self):
        self.license_auditor = LicenseAuditor(MODEL_LICENSE_CATALOG)

        # Registered factories / classes
        self.detectors: Dict[str, BasePedestrianDetector] = {
            "yolo": YOLODetectorAdapter(model_name="yolov8n.pt"),
            "yolo11x": YOLODetectorAdapter(model_name="yolo11x.pt"),
            "yolov8x": YOLODetectorAdapter(model_name="yolov8x.pt"),
            "rtdetr": RTDETRDetectorAdapter(model_name="rtdetr-l.pt"),
            "rtdetr_x": RTDETRDetectorAdapter(model_name="rtdetr-x.pt"),
            "fullbody_pose": FullBodyPoseDetectorAdapter(model_name="yolov8n-pose.pt"),
            "ensemble": EnsemblePedestrianDetector()
        }
        self.trackers: Dict[str, BaseMOTTracker] = {
            "bytetrack": ByteTrackAdapter(),
            "botsort": BoTSORTAdapter(),
            "deepstream": DeepStreamTrackerAdapter(),
            "mmtracking": MMTrackingAdapter()
        }
        self.reid_models: Dict[str, BasePersonReIDModel] = {
            "osnet": OSNetReIDAdapter(model_name="osnet_x1_0"),
            "fastreid": FastReIDAdapter(model_name="fastreid_sbs_r50")
        }
        self.face_analyzers: Dict[str, BaseCCTVFaceAnalyzer] = {
            "insightface": InsightFaceArcFaceAdapter(model_name="buffalo_l"),
            "sface": InsightFaceArcFaceAdapter(model_name="sface")
        }
        self.pose_estimators: Dict[str, BaseSkeletalPoseEstimator] = {
            "rtmpose": RTMPoseAdapter(model_name="rtmpose-m"),
            "mmpose": MMPoseAdapter(model_name="human")
        }
        self.gait_models: Dict[str, GaitAdapter] = {
            "handcrafted_kinematics": HandcraftedKinematicsAdapter(),
            "opengait": OpenGaitAdapter(model_name="GaitBase"),
            "gaitset": GaitSetAdapter(model_name="gaitset_cctv_v1")
        }


        # Active selections (default optimal configuration)
        self.active_keys = {
            "detection": "yolo",
            "tracking": "bytetrack",
            "reid": "osnet",
            "face": "insightface",
            "pose": "rtmpose",
            "gait": "gaitset"
        }

        self._load_yaml_config()

    def _load_yaml_config(self) -> None:
        """Load default active models from config/config.yaml if present."""
        import os
        import yaml
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cfg_path = os.path.join(base_dir, "config", "config.yaml")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                models_cfg = cfg.get("models", {})
                for cat in ["detection", "tracking", "reid", "face", "pose", "gait"]:
                    if cat in models_cfg and isinstance(models_cfg[cat], dict) and "active" in models_cfg[cat]:
                        target = str(models_cfg[cat]["active"]).lower().strip()
                        self.active_keys[cat] = target
            except Exception:
                pass

    # Accessors for active models
    @property
    def detector(self) -> BasePedestrianDetector:
        return self.detectors[self.active_keys["detection"]]

    @property
    def tracker(self) -> BaseMOTTracker:
        return self.trackers[self.active_keys["tracking"]]

    @property
    def reid(self) -> BasePersonReIDModel:
        return self.reid_models[self.active_keys["reid"]]

    @property
    def face(self) -> BaseCCTVFaceAnalyzer:
        return self.face_analyzers[self.active_keys["face"]]

    @property
    def pose(self) -> BaseSkeletalPoseEstimator:
        return self.pose_estimators[self.active_keys["pose"]]

    @property
    def gait(self) -> GaitAdapter:
        return self.gait_models[self.active_keys["gait"]]

    def select_model(self, category: str, model_id: str) -> Dict[str, Any]:
        """Dynamically switch the active model for a specific modality."""
        cat = category.lower().strip()
        mid = model_id.lower().strip()

        registry_map = {
            "detection": self.detectors,
            "tracking": self.trackers,
            "reid": self.reid_models,
            "face": self.face_analyzers,
            "pose": self.pose_estimators,
            "gait": self.gait_models
        }

        if cat not in registry_map:
            raise ValueError(f"Unknown modality category: '{cat}'. Valid categories: {list(registry_map.keys())}")

        available = registry_map[cat]
        if mid not in available:
            raise ValueError(f"Unknown model '{mid}' for '{cat}'. Available options: {list(available.keys())}")

        self.active_keys[cat] = mid
        info = self.license_auditor.get_info(mid)

        return {
            "status": "SUCCESS",
            "category": cat,
            "active_model": mid,
            "backend": available[mid].get_backend_info(),
            "license": {
                "code_license": info.code_license,
                "weights_license": info.weights_license,
                "is_commercial_ready": info.is_commercial_ready,
                "notice": info.restriction_notice
            }
        }

    def get_status(self) -> Dict[str, Any]:
        """Return the current active models, backends, and licensing overview."""
        audit = self.license_auditor.audit_active_stack(self.active_keys)

        return {
            "active_models": self.active_keys,
            "backends": {
                "detection": self.detector.get_backend_info(),
                "tracking": self.tracker.get_backend_info(),
                "reid": self.reid.get_backend_info(),
                "face": self.face.get_backend_info(),
                "pose": self.pose.get_backend_info(),
                "gait": self.gait.get_backend_info()
            },
            "compliance_audit": audit
        }

    def get_all_registered_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return list of all registered models across categories with license status."""
        catalog = {}

        mapping = {
            "detection": self.detectors,
            "tracking": self.trackers,
            "reid": self.reid_models,
            "face": self.face_analyzers,
            "pose": self.pose_estimators,
            "gait": self.gait_models
        }

        for cat, models_dict in mapping.items():
            cat_list = []
            for mid, inst in models_dict.items():
                lic = self.license_auditor.get_info(mid)
                cat_list.append({
                    "id": mid,
                    "name": lic.component_name,
                    "is_active": self.active_keys[cat] == mid,
                    "backend_info": inst.get_backend_info(),
                    "code_license": lic.code_license,
                    "weights_license": lic.weights_license,
                    "is_commercial_ready": lic.is_commercial_ready,
                    "restriction_notice": lic.restriction_notice,
                    "repo": lic.official_repo
                })
            catalog[cat] = cat_list

        return catalog


# Global singleton instance
model_registry = ModelRegistry()
