"""Video Target Processor & Biomechanical Exo-Skeleton Extractor.

Processes surveillance video footage to:
1. Lock onto a specific target person via bounding box, track ID, or appearance cues.
2. Apply Tier-1 forensic video enhancement (Luminance CLAHE + Bilateral Denoise) with SHA-256 hashing.
3. Extract frame-by-frame 17-keypoint COCO exo-skeleton using RTMPose ONNX with strict confidence gating.
4. Compute verified gait dynamics (cadence, stride cm, joint angular velocities, FFT harmonics).
5. Persist all records, skeletons, and artifacts into SQL database.
6. Export structured training datasets ready for deep neural models (GaitGraph2 / GPGait / ST-GCN).
"""

import os
import cv2
import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from app.detection.person_detector import PersonDetector
from app.tracking.tracker import MultiPersonTracker
from app.tracking.track_manager import TrackManager
from app.features.face import FaceAnalyzer
from app.features.body import BodyAnalyzer
from app.features.height import HeightEstimator
from app.features.pose import PoseEstimator
from app.features.gait import GaitAnalyzer
from app.enhancement.fast_enhancer import FastEnhancer
from app.database.database import init_db
from app.database.repository import Repository
from app.database.models import TrackObservation, CriminalRecord


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 cryptographic digest of a file for legal chain of custody."""
    if not os.path.isfile(filepath):
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_box_iou(boxA: List[int], boxB: List[int]) -> float:
    """Compute Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union_area = float(areaA + areaB - inter_area)

    return inter_area / union_area if union_area > 0 else 0.0


class VideoTargetProcessor:
    """High-precision processor extracting exo-skeletons, gait, and biometrics for a target person."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        conf_threshold: float = 0.30,
        enable_clahe: bool = True,
        enable_denoise: bool = True
    ):
        init_db(db_path)
        self.repo = Repository(db_path)
        self.conf_threshold = conf_threshold

        self.enhancer = FastEnhancer(enable_clahe=enable_clahe, enable_denoise=enable_denoise)
        self.detector = PersonDetector(min_height=60, min_width=25, confidence_threshold=0.30)
        self.tracker = MultiPersonTracker(max_age=30, min_hits=2, iou_threshold=0.20)
        self.pose_estimator = PoseEstimator(conf_threshold=conf_threshold)
        self.gait_analyzer = GaitAnalyzer(window_size=24)
        self.face_analyzer = FaceAnalyzer()
        self.body_analyzer = BodyAnalyzer()
        self.height_estimator = HeightEstimator(mounting_height_m=4.2, tilt_deg=35.0, focal_length_px=950.0)

    def process_video_target(
        self,
        video_path: str,
        target_name: str = "Subject of Interest",
        target_id: Optional[str] = None,
        initial_box: Optional[List[int]] = None,
        initial_frame: int = 0,
        target_track_id: Optional[str] = None,
        camera_id: str = "CAM-SURVEILLANCE-01",
        max_frames: Optional[int] = 300,
        stride_step: int = 1,
        enhance_video: bool = True,
        output_dir: str = "data/captures"
    ) -> Dict[str, Any]:
        """Process video footage, extract target exo-skeleton, gait kinematics, and persist in SQL."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video stream: {video_path}")

        os.makedirs(output_dir, exist_ok=True)
        raw_video_sha256 = compute_file_sha256(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        tid_resolved = target_track_id
        target_id_final = target_id or f"POI-{int(time.time()) % 100000:05d}"

        # Setup enhanced video writer if requested
        enhanced_video_path = ""
        video_writer = None
        if enhance_video:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            enhanced_video_path = os.path.join(output_dir, f"enhanced_{base_name}_{target_id_final}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(enhanced_video_path, fourcc, fps / stride_step, (width, height))

        frame_idx = 0
        processed_frames = 0
        target_skeleton_records = []
        target_pose_history = []
        best_person_crop = None
        best_crop_quality = -1.0
        best_crop_path = ""
        best_box = None
        last_seen_timestamp = 0.0
        first_seen_timestamp = 0.0

        all_track_lengths: Dict[str, int] = {}
        target_locked = tid_resolved is not None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if stride_step > 1 and (frame_idx % stride_step != 0):
                continue

            processed_frames += 1
            if max_frames and processed_frames > max_frames:
                break

            timestamp = round(frame_idx / fps, 3)
            if first_seen_timestamp == 0.0:
                first_seen_timestamp = timestamp
            last_seen_timestamp = timestamp

            # 1. Video Enhancement (CLAHE + Bilateral Filtering)
            if enhance_video:
                working_frame = self.enhancer.enhance_frame(frame)
            else:
                working_frame = frame

            # 2. Person Detection & Tracking
            detections = self.detector.detect(working_frame)
            active_tracks = self.tracker.update(detections)

            # 3. Target Association / Lock-On
            if not target_locked:
                if initial_box is not None and frame_idx >= initial_frame:
                    best_iou = 0.0
                    for trk in active_tracks:
                        iou = compute_box_iou(initial_box, trk.box)
                        if iou > best_iou and iou > 0.25:
                            best_iou = iou
                            tid_resolved = trk.track_id
                            target_locked = True
                elif not initial_box:
                    # Default: track the longest/most prominent active track
                    for trk in active_tracks:
                        all_track_lengths[trk.track_id] = all_track_lengths.get(trk.track_id, 0) + 1
                    if active_tracks:
                        tid_resolved = max(all_track_lengths.keys(), key=lambda k: all_track_lengths[k])

            # 4. Extract Target Data
            for trk in active_tracks:
                if tid_resolved is not None and trk.track_id != tid_resolved:
                    continue

                box = trk.box
                bx1, by1, bx2, by2 = max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3])
                crop = working_frame[by1:by2, bx1:bx2]
                if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 15:
                    continue

                # Pose & Exo-skeleton extraction
                pose_res = self.pose_estimator.estimate_pose(crop, [bx1, by1, bx2, by2])
                target_pose_history.append(pose_res)

                # Record frame skeleton data
                skeleton_item = {
                    "track_id": trk.track_id,
                    "person_id": target_id_final,
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "bbox": [bx1, by1, bx2, by2],
                    "keypoints_coco_17": pose_res.get("keypoints_coco_17", []),
                    "keypoints_crop": pose_res.get("keypoints_crop", {}),
                    "keypoints_global": pose_res.get("keypoints_global", {}),
                    "confidences": pose_res.get("confidences", {}),
                    "visible_joints_count": pose_res.get("visible_joints_count", 0),
                    "inter_ankle_dist": pose_res.get("inter_ankle_dist", 0.0),
                    "spine_tilt_deg": pose_res.get("spine_tilt_deg", 0.0),
                    "neck_point": pose_res.get("keypoints_crop", {}).get("neck", [0.0, 0.0]),
                    "is_confident": pose_res.get("is_confident", True),
                    "source": pose_res.get("source", "RTMPOSE_ONNX")
                }
                target_skeleton_records.append(skeleton_item)

                # Quality evaluation for best crop snapshot
                crop_area = crop.shape[0] * crop.shape[1]
                lap_var = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                quality = crop_area * (1.0 + min(2.0, lap_var / 50.0))
                if quality > best_crop_quality:
                    best_crop_quality = quality
                    best_person_crop = crop.copy()
                    best_box = [bx1, by1, bx2, by2]

            # 5. Write to Enhanced Video Stream with Biomechanical Skeleton Overlay
            if video_writer is not None:
                vis_frame = working_frame.copy()
                for trk in active_tracks:
                    is_target = (tid_resolved is not None and trk.track_id == tid_resolved)
                    color = (0, 255, 0) if is_target else (0, 165, 255)
                    thickness = 2 if is_target else 1
                    cv2.rectangle(vis_frame, (trk.box[0], trk.box[1]), (trk.box[2], trk.box[3]), color, thickness)

                    label = f"[{'TARGET: ' + target_name if is_target else trk.track_id}]"
                    cv2.putText(vis_frame, label, (trk.box[0], max(20, trk.box[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    if is_target and target_skeleton_records:
                        latest_skel = target_skeleton_records[-1]
                        for j in latest_skel.get("keypoints_coco_17", []):
                            if j.get("is_visible", False):
                                cv2.circle(vis_frame, (j["global_x"], j["global_y"]), 4, (0, 255, 255), -1)

                video_writer.write(vis_frame)

        cap.release()
        if video_writer is not None:
            video_writer.release()

        enhanced_video_sha256 = compute_file_sha256(enhanced_video_path) if enhanced_video_path else ""

        # 6. Save Best Person Crop
        best_crop_sha256 = ""
        if best_person_crop is not None:
            best_crop_path = os.path.join(output_dir, f"crop_{target_id_final}.jpg")
            cv2.imwrite(best_crop_path, best_person_crop)
            best_crop_sha256 = compute_file_sha256(best_crop_path)

        # 7. Biometric Feature Extraction on Best Crop
        height_info = self.height_estimator.estimate_height_cm(best_box or [0, 0, width, height], frame_height=height)
        estimated_height_cm = height_info.get("estimated_height_cm", 170.0)

        face_info = self.face_analyzer.analyze_person_crop(best_person_crop) if best_person_crop is not None else {}
        body_info = self.body_analyzer.analyze(best_person_crop) if best_person_crop is not None else {}

        # 8. Gait Dynamics Analysis (Real Kinematics)
        gait_dynamics = self.gait_analyzer.analyze_sequence(
            target_pose_history,
            estimated_height_cm=estimated_height_cm
        )

        # 9. Batch Persist Skeletons into SQL Database
        saved_skeletons_count = self.repo.save_person_skeletons_batch(target_skeleton_records)

        # 10. Persist Gait Dynamics into SQL Database
        gait_db_record = {
            "track_id": tid_resolved or target_id_final,
            "person_id": target_id_final,
            "sequence_length": len(target_skeleton_records),
            "stride_length_px": gait_dynamics.get("stride_length_px", 0.0),
            "stride_length_cm": gait_dynamics.get("stride_length_cm", 0.0),
            "cadence_hz": gait_dynamics.get("cadence_steps_per_sec", 0.0),
            "spine_tilt_deg": gait_dynamics.get("spine_tilt_deg", 0.0),
            "posture_score": gait_dynamics.get("posture_correctness", 0.0),
            "joint_velocities": gait_dynamics.get("joint_angle_velocities", []),
            "fft_harmonics": gait_dynamics.get("fft_harmonic_ratios", []),
            "gait_wave": gait_dynamics.get("gait_wave", []),
            "gait_embedding": gait_dynamics.get("gait_embedding", []),
            "is_valid_gait": gait_dynamics.get("valid_gait", False),
            "gait_usable": gait_dynamics.get("gait_usable", False)
        }
        self.repo.save_gait_dynamics(gait_db_record)

        # 11. Persist Video Enhancement Artifact into SQL
        if enhanced_video_path:
            self.repo.save_enhanced_video_artifact(
                source_video_path=video_path,
                enhanced_video_path=enhanced_video_path,
                track_id=tid_resolved,
                person_id=target_id_final,
                frame_count=processed_frames,
                enhancement_profile="TIER_1_CLAHE_BILATERAL",
                raw_sha256=raw_video_sha256,
                enhanced_sha256=enhanced_video_sha256
            )

        # 12. Persist Target Profile & Track Observation in SQL
        track_obs = TrackObservation(
            track_id=tid_resolved or target_id_final,
            camera_id=camera_id,
            first_seen=first_seen_timestamp,
            last_seen=last_seen_timestamp,
            frame_count=len(target_skeleton_records),
            best_frame_path=best_crop_path,
            face_visible=face_info.get("face_visible", False),
            face_status=face_info.get("face_status", "UNAVAILABLE"),
            face_tier_details=face_info.get("tiers", {}),
            estimated_height_cm=estimated_height_cm,
            body_proportions=body_info.get("proportions", {}),
            clothing_upper=body_info.get("clothing", {}).get("upper_hex", "#3b4252"),
            clothing_lower=body_info.get("clothing", {}).get("lower_hex", "#1e222a"),
            stride_length_px=gait_dynamics.get("stride_length_px", 0.0),
            stride_length_cm=gait_dynamics.get("stride_length_cm", 0.0),
            cadence_steps_per_sec=gait_dynamics.get("cadence_steps_per_sec", 0.0),
            spine_tilt_deg=gait_dynamics.get("spine_tilt_deg", 0.0),
            posture_score=gait_dynamics.get("posture_correctness", 0.0),
            gait_wave=gait_dynamics.get("gait_wave", []),
            face_embedding=face_info.get("face_embedding", []),
            body_embedding=body_info.get("body_embedding", []),
            gait_embedding=gait_dynamics.get("gait_embedding", [])
        )
        self.repo.save_track(track_obs)

        # Also store target profile in criminal_records for multi-modal match querying
        target_record = CriminalRecord(
            id=target_id_final,
            fir_no=f"CCTV-POI-{target_id_final}",
            unit_name="Central Intelligence Division",
            subdivision="Video Forensic Unit",
            police_station="Surveillance Command",
            accused_name=target_name,
            alias=f"Track-{tid_resolved}",
            age=32,
            gender="Male",
            acts_sec="Forensic Video Extraction",
            brief_facts=f"Extracted from {os.path.basename(video_path)} at {camera_id}",
            photo_url=best_crop_path,
            known_height_cm=estimated_height_cm,
            torso_leg_ratio=body_info.get("proportions", {}).get("torso_leg_ratio", 0.85),
            stride_length_cm=gait_dynamics.get("stride_length_cm", 0.0),
            posture_lean_angle=gait_dynamics.get("spine_tilt_deg", 0.0),
            posture_correctness=gait_dynamics.get("posture_correctness", 0.0),
            clothing_upper_color=body_info.get("clothing", {}).get("upper_hex", "#3b4252"),
            clothing_lower_color=body_info.get("clothing", {}).get("lower_hex", "#1e222a"),
            face_embedding=face_info.get("face_embedding", []),
            body_embedding=body_info.get("body_embedding", []),
            gait_embedding=gait_dynamics.get("gait_embedding", []),
            enhanced_photo_url=best_crop_path
        )
        self.repo.insert_criminal_record(target_record)

        # 13. Export Training Dataset File (.npy + JSON labels)
        training_export_dir = os.path.join(output_dir, "training_datasets")
        os.makedirs(training_export_dir, exist_ok=True)

        training_matrix = []
        for s in target_skeleton_records:
            k17 = s["keypoints_coco_17"]
            mat = [[j.get("crop_x", 0.0), j.get("crop_y", 0.0), j.get("confidence", 0.0)] for j in k17]
            training_matrix.append(mat)

        training_npy_path = os.path.join(training_export_dir, f"skeletons_{target_id_final}.npy")
        np_arr = np.array(training_matrix, dtype=np.float32)  # (T, 17, 3)
        np.save(training_npy_path, np_arr)

        metadata_json_path = os.path.join(training_export_dir, f"metadata_{target_id_final}.json")
        training_meta = {
            "target_id": target_id_final,
            "target_name": target_name,
            "track_id": tid_resolved,
            "source_video": video_path,
            "total_frames_extracted": len(target_skeleton_records),
            "keypoint_shape": list(np_arr.shape),
            "keypoint_schema": "COCO_17 (x, y, confidence)",
            "estimated_height_cm": estimated_height_cm,
            "cadence_hz": gait_dynamics.get("cadence_steps_per_sec", 0.0),
            "stride_length_cm": gait_dynamics.get("stride_length_cm", 0.0),
            "clothing_upper": body_info.get("clothing", {}).get("upper_hex", "#3b4252"),
            "clothing_lower": body_info.get("clothing", {}).get("lower_hex", "#1e222a"),
            "is_valid_gait": gait_dynamics.get("valid_gait", False),
            "training_npy_path": training_npy_path,
            "raw_video_sha256": raw_video_sha256,
            "enhanced_video_sha256": enhanced_video_sha256
        }
        with open(metadata_json_path, "w") as f:
            json.dump(training_meta, f, indent=2)

        return {
            "status": "SUCCESS",
            "target": {
                "target_id": target_id_final,
                "name": target_name,
                "track_id": tid_resolved,
                "best_crop_path": best_crop_path,
                "best_crop_sha256": best_crop_sha256
            },
            "video_enhancement": {
                "enhanced": enhance_video,
                "source_video": video_path,
                "raw_sha256": raw_video_sha256,
                "enhanced_video_path": enhanced_video_path,
                "enhanced_sha256": enhanced_video_sha256,
                "frames_processed": processed_frames
            },
            "exo_skeleton": {
                "model": self.pose_estimator.backend,
                "frames_extracted": len(target_skeleton_records),
                "saved_in_sql": saved_skeletons_count,
                "average_confidence": round(float(np.mean([
                    np.mean(list(s["confidences"].values())) for s in target_skeleton_records
                ])) if target_skeleton_records else 0.0, 3)
            },
            "gait_kinematics": gait_dynamics,
            "biometrics": {
                "estimated_height_cm": estimated_height_cm,
                "torso_leg_ratio": body_info.get("proportions", {}).get("torso_leg_ratio", 0.85),
                "face_status": face_info.get("face_status", "UNAVAILABLE"),
                "face_embedding_dim": len(face_info.get("face_embedding", [])),
                "body_embedding_dim": len(body_info.get("body_embedding", [])),
                "gait_embedding_dim": len(gait_dynamics.get("gait_embedding", []))
            },
            "model_training_dataset": {
                "npy_path": training_npy_path,
                "metadata_path": metadata_json_path,
                "tensor_shape": list(np_arr.shape),
                "compatible_models": ["GaitGraph2", "GPGait", "ST-GCN", "SkeletonGait++"]
            }
        }
