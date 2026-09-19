"""Pose and skeletal landmark estimation module for CCTV gait and posture analysis."""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple

JOINT_NAMES = [
    "nose", "neck",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle"
]

class PoseEstimator:
    def __init__(self):
        pass

    def estimate_pose(self, person_crop: np.ndarray, box: List[int]) -> Dict[str, Any]:
        """Estimate 14 skeletal landmarks in both crop and global frame coordinates."""
        if person_crop is None or person_crop.size == 0:
            return {"keypoints": {}, "visible": False}

        ph, pw = person_crop.shape[:2]
        x1, y1 = box[0], box[1]

        # Extract silhouette via thresholding / background contrast
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 1. Head and Neck
        head_x = pw * 0.5
        head_y = ph * 0.06
        neck_x = pw * 0.5
        neck_y = ph * 0.16

        # 2. Shoulders
        sh_y = ph * 0.20
        # Find horizontal edges at shoulder height
        row_sh = thresh[int(sh_y), :] if int(sh_y) < ph else np.array([])
        nonzero_sh = np.where(row_sh > 0)[0]
        if len(nonzero_sh) > 4:
            left_sh_x = float(nonzero_sh[0])
            right_sh_x = float(nonzero_sh[-1])
        else:
            left_sh_x = pw * 0.25
            right_sh_x = pw * 0.75

        # 3. Hips
        hip_y = ph * 0.52
        left_hip_x = pw * 0.35
        right_hip_x = pw * 0.65

        # 4. Legs & Ankles (Crucial for Stride & Gait Dynamics!)
        # Inspect lower 25% of silhouette to detect foot separation
        lower_region = thresh[int(ph * 0.75):int(ph * 0.98), :]
        if lower_region.size > 0:
            col_sums = np.sum(lower_region > 0, axis=0)
            foot_cols = np.where(col_sums > 3)[0]
            if len(foot_cols) > 6:
                # Two distinct foot clusters or stride separation
                left_ankle_x = float(foot_cols[0] + 2)
                right_ankle_x = float(foot_cols[-1] - 2)
            else:
                left_ankle_x = pw * 0.38
                right_ankle_x = pw * 0.62
        else:
            left_ankle_x = pw * 0.38
            right_ankle_x = pw * 0.62

        left_ankle_y = ph * 0.95
        right_ankle_y = ph * 0.95

        # Knees interpolated between hips and ankles
        left_knee_x = (left_hip_x + left_ankle_x) / 2.0
        left_knee_y = ph * 0.74
        right_knee_x = (right_hip_x + right_ankle_x) / 2.0
        right_knee_y = ph * 0.74

        # Arms / Elbows / Wrists
        left_elbow_x = left_sh_x - (pw * 0.08)
        left_elbow_y = ph * 0.36
        left_wrist_x = left_elbow_x - (pw * 0.04)
        left_wrist_y = ph * 0.48

        right_elbow_x = right_sh_x + (pw * 0.08)
        right_elbow_y = ph * 0.36
        right_wrist_x = right_elbow_x + (pw * 0.04)
        right_wrist_y = ph * 0.48

        keypoints_crop = {
            "nose": [round(head_x, 1), round(head_y, 1)],
            "neck": [round(neck_x, 1), round(neck_y, 1)],
            "right_shoulder": [round(right_sh_x, 1), round(sh_y, 1)],
            "right_elbow": [round(right_elbow_x, 1), round(right_elbow_y, 1)],
            "right_wrist": [round(right_wrist_x, 1), round(right_wrist_y, 1)],
            "left_shoulder": [round(left_sh_x, 1), round(sh_y, 1)],
            "left_elbow": [round(left_elbow_x, 1), round(left_elbow_y, 1)],
            "left_wrist": [round(left_wrist_x, 1), round(left_wrist_y, 1)],
            "right_hip": [round(right_hip_x, 1), round(hip_y, 1)],
            "right_knee": [round(right_knee_x, 1), round(right_knee_y, 1)],
            "right_ankle": [round(right_ankle_x, 1), round(right_ankle_y, 1)],
            "left_hip": [round(left_hip_x, 1), round(hip_y, 1)],
            "left_knee": [round(left_knee_x, 1), round(left_knee_y, 1)],
            "left_ankle": [round(left_ankle_x, 1), round(left_ankle_y, 1)],
        }

        # Compute global coordinates for canvas / video overlay
        keypoints_global = {}
        for k, (kx, ky) in keypoints_crop.items():
            keypoints_global[k] = [int(x1 + kx), int(y1 + ky)]

        # Calculate stride separation in pixels
        inter_ankle_dist = abs(right_ankle_x - left_ankle_x)

        return {
            "keypoints_crop": keypoints_crop,
            "keypoints_global": keypoints_global,
            "inter_ankle_dist": round(inter_ankle_dist, 1),
            "visible": True
        }
