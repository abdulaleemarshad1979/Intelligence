"""Open-Source Code & Model License Audit System for CCTV Intelligence.

Audits licenses across all integrated candidates:
- OpenGait (Academic/Non-Commercial restriction)
- InsightFace (Code MIT, Pretrained Weights Non-Commercial Research restriction)
- NVIDIA DeepStream (Apache-2.0 commercial ready)
- FastReID (Apache-2.0 commercial ready)
- Torchreid / OSNet (MIT commercial ready)
- MMPose / RTMPose (Apache-2.0 commercial ready)
- MMTracking (Apache-2.0 commercial ready)
- YOLO (AGPL-3.0 copyleft notice)
- RT-DETR (Apache-2.0 commercial ready)
- ByteTrack / BoT-SORT (MIT commercial ready)
"""

from typing import Dict, Any, List
from app.adapters.base import ModelLicenseInfo


MODEL_LICENSE_CATALOG: Dict[str, ModelLicenseInfo] = {
    # Detection
    "yolo": ModelLicenseInfo(
        model_id="yolo",
        component_name="Ultralytics YOLO (v8 / 11)",
        capability="Person Detection",
        code_license="AGPL-3.0 / Commercial",
        weights_license="AGPL-3.0 / Commercial",
        is_commercial_ready=False,
        restriction_notice="AGPL-3.0 copyleft requires open-sourcing derivative backend services or purchasing an Ultralytics Commercial License for closed-source deployment.",
        official_repo="https://github.com/ultralytics/ultralytics"
    ),
    "rtdetr": ModelLicenseInfo(
        model_id="rtdetr",
        component_name="RT-DETR (Real-Time DEtection TRansformer)",
        capability="Person Detection",
        code_license="Apache-2.0",
        weights_license="Apache-2.0",
        is_commercial_ready=True,
        restriction_notice="Permissive Apache-2.0 license. Fully cleared for commercial enterprise and public law-enforcement deployment without copyleft constraints.",
        official_repo="https://github.com/lyuwenyu/RT-DETR"
    ),
    # Tracking
    "bytetrack": ModelLicenseInfo(
        model_id="bytetrack",
        component_name="ByteTrack Multi-Object Tracker",
        capability="MOT Association",
        code_license="MIT",
        weights_license="N/A (Heuristic / Motion Association)",
        is_commercial_ready=True,
        restriction_notice="Permissive MIT license. Freely usable in commercial and law-enforcement operations.",
        official_repo="https://github.com/ifzhang/ByteTrack"
    ),
    "botsort": ModelLicenseInfo(
        model_id="botsort",
        component_name="BoT-SORT Tracker (with Camera Motion Compensation)",
        capability="MOT Association + CMC",
        code_license="MIT",
        weights_license="MIT",
        is_commercial_ready=True,
        restriction_notice="Permissive MIT license with robust camera motion handling. Commercial ready.",
        official_repo="https://github.com/NirAharon/BoT-SORT"
    ),
    "deepstream": ModelLicenseInfo(
        model_id="deepstream",
        component_name="NVIDIA DeepStream SDK / Gst-nvtracker",
        capability="Multi-Camera Video Pipeline & Tracking",
        code_license="Apache-2.0",
        weights_license="Proprietary / TensorRT",
        is_commercial_ready=True,
        restriction_notice="Apache-2.0 open-source plugins with NVIDIA SLA. Ideal for dense multi-camera hardware acceleration.",
        official_repo="https://github.com/NVIDIA/DeepStream"
    ),
    "mmtracking": ModelLicenseInfo(
        model_id="mmtracking",
        component_name="OpenMMLab MMTracking",
        capability="Unified Video Perception & MOT",
        code_license="Apache-2.0",
        weights_license="Apache-2.0",
        is_commercial_ready=True,
        restriction_notice="Permissive Apache-2.0 framework for multi-object tracking and video segmentation.",
        official_repo="https://github.com/open-mmlab/mmtracking"
    ),
    # Re-ID
    "osnet": ModelLicenseInfo(
        model_id="osnet",
        component_name="Torchreid OSNet (Omni-Scale Network)",
        capability="Person Re-Identification",
        code_license="MIT",
        weights_license="MIT",
        is_commercial_ready=True,
        restriction_notice="Permissive MIT license. Weights and code cleared for commercial deployment.",
        official_repo="https://github.com/KaiyangZhou/deep-person-reid"
    ),
    "fastreid": ModelLicenseInfo(
        model_id="fastreid",
        component_name="FastReID (SBS-50 / BoT / AGW / TransReID)",
        capability="Person Re-Identification",
        code_license="Apache-2.0",
        weights_license="Apache-2.0",
        is_commercial_ready=True,
        restriction_notice="Apache-2.0 research & production framework. Highly modular with diverse Re-ID backbones.",
        official_repo="https://github.com/JDAI-CV/FastReID"
    ),
    # Face
    "insightface": ModelLicenseInfo(
        model_id="insightface",
        component_name="InsightFace / ArcFace (3-Tier CCTV Adaptation)",
        capability="Face Detection & ArcFace Embedding",
        code_license="MIT",
        weights_license="Non-Commercial Research Only",
        is_commercial_ready=False,
        restriction_notice="CRITICAL: InsightFace source code is MIT, but public pretrained recognition model weights (buffalo_l, antelopev2) are restricted to non-commercial research. Commercial deployment requires separate commercial licensing or in-house retrained weights.",
        official_repo="https://github.com/deepinsight/insightface"
    ),
    "sface": ModelLicenseInfo(
        model_id="sface",
        component_name="OpenCV Zoo (YuNet + SFace)",
        capability="Face Detection & SFace Recognition",
        code_license="Apache-2.0",
        weights_license="Apache-2.0",
        is_commercial_ready=True,
        restriction_notice="Permissive Apache-2.0 / BSD license from OpenCV Model Zoo. Fully cleared for commercial enterprise and public safety operations without academic-only restrictions.",
        official_repo="https://github.com/opencv/opencv_zoo"
    ),
    # Pose
    "rtmpose": ModelLicenseInfo(
        model_id="rtmpose",
        component_name="OpenMMLab RTMPose",
        capability="Real-Time 17-Keypoint Pose Estimation",
        code_license="Apache-2.0",
        weights_license="Apache-2.0",
        is_commercial_ready=True,
        restriction_notice="Permissive Apache-2.0 license. Optimized for high-FPS edge & server inference.",
        official_repo="https://github.com/open-mmlab/mmpose"
    ),
    "mmpose": ModelLicenseInfo(
        model_id="mmpose",
        component_name="OpenMMLab MMPose",
        capability="Human Keypoint Estimation",
        code_license="Apache-2.0",
        weights_license="Apache-2.0",
        is_commercial_ready=True,
        restriction_notice="Permissive Apache-2.0 license. Rich multi-model pose estimation library.",
        official_repo="https://github.com/open-mmlab/mmpose"
    ),
    # Gait
    "opengait": ModelLicenseInfo(
        model_id="opengait",
        component_name="OpenGait Research Framework (GaitBase / GaitGL)",
        capability="Cross-View Gait Sequence Recognition",
        code_license="Academic Non-Commercial Only",
        weights_license="Academic Non-Commercial Only",
        is_commercial_ready=False,
        restriction_notice="CRITICAL: OpenGait code and models are strictly restricted to academic research purposes. Production law-enforcement deployment requires permission or using our native GaitSet kinematics pipeline.",
        official_repo="https://github.com/ShiqiYu/OpenGait"
    ),
    "gaitset": ModelLicenseInfo(
        model_id="gaitset",
        component_name="GaitSet Temporal Set-Pooling & Kinematics",
        capability="Temporal Gait Waveform & Cadence Dynamics",
        code_license="MIT",
        weights_license="MIT / Clean In-House",
        is_commercial_ready=True,
        restriction_notice="Permissive MIT implementation with in-house kinematic stride/cadence math. Fully cleared for commercial operations.",
        official_repo="https://github.com/AbnerHqC/GaitSet"
    )
}


class LicenseAuditor:
    """Automated license auditor assessing commercial law-enforcement compliance."""

    def __init__(self, catalog: Dict[str, ModelLicenseInfo] = MODEL_LICENSE_CATALOG):
        self.catalog = catalog

    def get_info(self, model_id: str) -> ModelLicenseInfo:
        key = model_id.lower().strip()
        return self.catalog.get(key, ModelLicenseInfo(
            model_id=key,
            component_name=key.upper(),
            capability="Perception",
            code_license="Unknown",
            weights_license="Unknown",
            is_commercial_ready=False,
            restriction_notice="Unverified third-party component. Manual legal audit required before deployment.",
            official_repo=""
        ))

    def audit_active_stack(self, active_selections: Dict[str, str]) -> Dict[str, Any]:
        """Audit the currently chosen model stack for licensing compliance."""
        total = 0
        commercial_ready_count = 0
        academic_restricted: List[Dict[str, Any]] = []
        copyleft_flagged: List[Dict[str, Any]] = []
        fully_cleared: List[Dict[str, Any]] = []

        for category, model_id in active_selections.items():
            info = self.get_info(model_id)
            total += 1
            item = {
                "category": category,
                "model_id": info.model_id,
                "name": info.component_name,
                "code_license": info.code_license,
                "weights_license": info.weights_license,
                "is_commercial_ready": info.is_commercial_ready,
                "restriction_notice": info.restriction_notice,
                "repo": info.official_repo
            }
            if info.is_commercial_ready:
                commercial_ready_count += 1
                fully_cleared.append(item)
            elif "Academic" in info.code_license or "Non-Commercial" in info.weights_license:
                academic_restricted.append(item)
            elif "AGPL" in info.code_license:
                copyleft_flagged.append(item)
            else:
                academic_restricted.append(item)

        overall_compliance = "RESEARCH_ONLY" if academic_restricted else ("COMMERCIAL_COPYLEFT_NOTICE" if copyleft_flagged else "COMMERCIAL_PRODUCTION_READY")

        return {
            "overall_status": overall_compliance,
            "total_components": total,
            "commercial_ready_count": commercial_ready_count,
            "academic_restricted_components": academic_restricted,
            "copyleft_flagged_components": copyleft_flagged,
            "fully_cleared_components": fully_cleared
        }

    def get_full_catalog(self) -> List[Dict[str, Any]]:
        return [
            {
                "model_id": m.model_id,
                "name": m.component_name,
                "capability": m.capability,
                "code_license": m.code_license,
                "weights_license": m.weights_license,
                "is_commercial_ready": m.is_commercial_ready,
                "restriction_notice": m.restriction_notice,
                "repo": m.official_repo
            }
            for m in self.catalog.values()
        ]
