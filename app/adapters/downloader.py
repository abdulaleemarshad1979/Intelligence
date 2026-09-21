"""Model Weights Downloader for CCTV Intelligence.

Downloads verified, official neural model checkpoints and ONNX representations directly
from official GitHub releases, OpenCV Zoo, and Hugging Face into the local `models/` directory.
"""

import os
import sys
import time
import urllib.request
from typing import Dict, Any, List, Optional, Callable


def get_models_dir() -> str:
    """Return the absolute path to the local models directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    m_dir = os.path.join(base_dir, "models")
    os.makedirs(m_dir, exist_ok=True)
    return m_dir


# Official GitHub & Hub Model Catalog
GITHUB_MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "face_yunet": {
        "model_id": "face_yunet",
        "name": "YuNet Face Detector (ONNX)",
        "category": "face",
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "size_bytes": 343209,
        "is_essential": True,
        "source": "OpenCV Zoo GitHub",
        "description": "Ultra-fast SOTA 5-landmark face detector running via cv2.FaceDetectorYN."
    },
    "face_sface": {
        "model_id": "face_sface",
        "name": "SFace Face Recognizer (ONNX)",
        "category": "face",
        "filename": "face_recognition_sface_2021dec.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "size_bytes": 38780775,
        "is_essential": True,
        "source": "OpenCV Zoo GitHub",
        "description": "128-d deep facial feature extractor running via cv2.FaceRecognizerSF."
    },
    "yolov8n": {
        "model_id": "yolov8n",
        "name": "YOLOv8 Nano Pedestrian Detector",
        "category": "detection",
        "filename": "yolov8n.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "size_bytes": 6251480,
        "is_essential": True,
        "source": "Ultralytics GitHub Releases",
        "description": "Real-time pedestrian bounding box detector with ByteTrack / BoT-SORT integration."
    },
    "yolov8n_pose": {
        "model_id": "yolov8n_pose",
        "name": "YOLOv8 Nano 17-Keypoint Pose Estimator",
        "category": "pose",
        "filename": "yolov8n-pose.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt",
        "size_bytes": 6788480,
        "is_essential": True,
        "source": "Ultralytics GitHub Releases",
        "description": "Real-time 17 skeletal joint estimator for gait cadence, stride, and posture lean."
    },
    "osnet": {
        "model_id": "osnet",
        "name": "Torchreid OSNet Omni-Scale Re-ID",
        "category": "reid",
        "filename": "osnet_x1_0_imagenet.pth",
        "url": "https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x1_0_imagenet.pth",
        "size_bytes": 8806280,
        "is_essential": True,
        "source": "KaiyangZhou / OSNet GitHub & HuggingFace",
        "description": "512-dimensional multi-scale body appearance embedding for person re-identification."
    },
    "rtdetr": {
        "model_id": "rtdetr",
        "name": "RT-DETR Transformer Detector",
        "category": "detection",
        "filename": "rtdetr-l.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-l.pt",
        "size_bytes": 67000000,
        "is_essential": False,
        "source": "Ultralytics / RT-DETR GitHub Releases",
        "description": "Real-time end-to-end detection transformer for high-accuracy crowded surveillance."
    },
    "rtdetr_x": {
        "model_id": "rtdetr_x",
        "name": "RT-DETR-X Transformer Detector (X-Large)",
        "category": "detection",
        "filename": "rtdetr-x.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-x.pt",
        "size_bytes": 134000000,
        "is_essential": False,
        "source": "Ultralytics / RT-DETR GitHub Releases",
        "description": "Top-tier SOTA detection transformer for extreme crowd occlusion."
    },
    "yolo11x": {
        "model_id": "yolo11x",
        "name": "YOLO11 X-Large Pedestrian Detector",
        "category": "detection",
        "filename": "yolo11x.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
        "size_bytes": 113000000,
        "is_essential": False,
        "source": "Ultralytics GitHub Releases",
        "description": "2024 flagship heavyweight detector with maximum pedestrian localization mAP."
    },
    "yolo11l": {
        "model_id": "yolo11l",
        "name": "YOLO11 Large Pedestrian Detector",
        "category": "detection",
        "filename": "yolo11l.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
        "size_bytes": 53000000,
        "is_essential": False,
        "source": "Ultralytics GitHub Releases",
        "description": "2024 SOTA large detector balancing speed and accuracy."
    },
    "yolov8x": {
        "model_id": "yolov8x",
        "name": "YOLOv8 X-Large Pedestrian Detector",
        "category": "detection",
        "filename": "yolov8x.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt",
        "size_bytes": 136000000,
        "is_essential": False,
        "source": "Ultralytics GitHub Releases",
        "description": "Heavyweight benchmark detector for high-resolution CCTV."
    },
    "yolo11x_pose": {
        "model_id": "yolo11x_pose",
        "name": "YOLO11 X-Large 17-Keypoint Pose Estimator",
        "category": "pose",
        "filename": "yolo11x-pose.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt",
        "size_bytes": 118000000,
        "is_essential": False,
        "source": "Ultralytics GitHub Releases",
        "description": "Top-tier 17-joint full body skeleton and posture dynamics estimator."
    }
}


class ModelDownloader:
    """Manages downloading, verifying, and checking neural weights in models/."""

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or get_models_dir()

    def get_model_path(self, model_key: str) -> Optional[str]:
        """Return the path if model file exists and is non-empty, else None."""
        if model_key not in GITHUB_MODEL_CATALOG:
            return None
        info = GITHUB_MODEL_CATALOG[model_key]
        p = os.path.join(self.models_dir, info["filename"])
        if os.path.isfile(p) and os.path.getsize(p) > 1024:
            return p
        return None

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Return download status of all cataloged models."""
        status = {}
        for key, info in GITHUB_MODEL_CATALOG.items():
            path = os.path.join(self.models_dir, info["filename"])
            is_present = os.path.isfile(path) and os.path.getsize(path) > 1024
            size = os.path.getsize(path) if is_present else 0

            status[key] = {
                "model_id": key,
                "name": info["name"],
                "category": info["category"],
                "filename": info["filename"],
                "is_downloaded": is_present,
                "path": path if is_present else None,
                "size_mb": round(size / (1024 * 1024), 2) if is_present else 0.0,
                "expected_size_mb": round(info["size_bytes"] / (1024 * 1024), 2),
                "is_essential": info["is_essential"],
                "source": info["source"],
                "description": info["description"]
            }
        return status

    def download_model(
        self,
        model_key: str,
        force: bool = False,
        progress_cb: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """Download model weights from GitHub release with streaming and atomic write."""
        if model_key not in GITHUB_MODEL_CATALOG:
            raise ValueError(f"Unknown model key '{model_key}'. Available: {list(GITHUB_MODEL_CATALOG.keys())}")

        info = GITHUB_MODEL_CATALOG[model_key]
        dest_path = os.path.join(self.models_dir, info["filename"])
        temp_path = dest_path + ".tmp"

        # If already exists and not forced, return immediately
        if not force and os.path.isfile(dest_path) and os.path.getsize(dest_path) > 1024:
            return {
                "status": "ALREADY_EXISTS",
                "model_id": model_key,
                "filename": info["filename"],
                "path": dest_path,
                "size_mb": round(os.path.getsize(dest_path) / (1024 * 1024), 2)
            }

        url = info["url"]
        headers = {"User-Agent": "CCTV-Intelligence/2.1 (Linux)"}
        req = urllib.request.Request(url, headers=headers)

        start_time = time.time()
        with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as f_out:
            total_size = int(resp.headers.get("Content-Length", info["size_bytes"]))
            downloaded = 0
            chunk_size = 128 * 1024  # 128 KB

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
                downloaded += len(chunk)
                if progress_cb:
                    progress_cb(downloaded, total_size)

        # Atomic replace
        os.replace(temp_path, dest_path)
        elapsed = time.time() - start_time
        final_size = os.path.getsize(dest_path)

        return {
            "status": "DOWNLOADED",
            "model_id": model_key,
            "filename": info["filename"],
            "path": dest_path,
            "size_mb": round(final_size / (1024 * 1024), 2),
            "elapsed_sec": round(elapsed, 2)
        }

    def download_essential_models(
        self,
        progress_cb: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, Any]:
        """Download all essential models needed for the full multi-modal pipeline."""
        results = {}
        for key, info in GITHUB_MODEL_CATALOG.items():
            if not info.get("is_essential", False):
                continue
            def cb(dl, tot, k=key):
                if progress_cb:
                    progress_cb(k, dl, tot)
            try:
                res = self.download_model(key, progress_cb=cb)
                results[key] = res
            except Exception as e:
                results[key] = {"status": "ERROR", "error": str(e)}
        return results


# Global singleton instance
model_downloader = ModelDownloader()
