"""
Synthetic ground-truth validation for height_estimation.py.

Builds a virtual camera with known pose, projects known-height virtual
people into it, recovers their pose via the same solvePnP path
field_calibration_tool.py uses, then checks that estimate_height_cm
recovers the true height. This isolates the *geometry math* from
real-world calibration/detection error, so a failure here means the
math itself is broken, not that a camera was mis-calibrated in the field.
"""
import json
import tempfile
import os

import cv2
import numpy as np
import pytest

from height_estimation import CameraCalibration, estimate_height_cm, ground_position_m


def make_synthetic_calibration(tmp_path, cam_height=4.2, tilt_deg=22.0, fx=1000.0, fy=1000.0, cx=960.0, cy=540.0):
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    R_base = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    t = np.deg2rad(tilt_deg)
    R_tilt = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]], dtype=np.float64)
    R_true = R_tilt @ R_base
    C_world = np.array([0.0, 0.0, cam_height])
    tvec_true = -R_true @ C_world

    def project(Xw):
        Xc = R_true @ np.asarray(Xw, dtype=np.float64) + tvec_true
        x = K @ Xc
        return x[:2] / x[2]

    ground_pts_world = np.array([[-2, 6, 0], [2, 6, 0], [2, 12, 0], [-2, 12, 0]], dtype=np.float64)
    ground_pts_img = np.array([project(p) for p in ground_pts_world], dtype=np.float64)

    ok, rvec_est, tvec_est = cv2.solvePnP(ground_pts_world, ground_pts_img, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    assert ok

    calib_path = os.path.join(tmp_path, "test_cam.json")
    with open(calib_path, "w") as f:
        json.dump({
            "camera_id": "TEST-CAM",
            "K": K.tolist(),
            "rvec": rvec_est.flatten().tolist(),
            "tvec": tvec_est.flatten().tolist(),
        }, f)
    return calib_path, project


@pytest.mark.parametrize("ground_pos,true_height_m", [
    ((0.0, 8.0), 1.75),
    ((-1.5, 10.0), 1.60),
    ((1.2, 7.0), 1.90),
    ((0.5, 15.0), 1.68),
])
def test_height_recovery_various_positions(tmp_path, ground_pos, true_height_m):
    calib_path, project = make_synthetic_calibration(tmp_path)
    calib = CameraCalibration(calib_path)

    foot_w = np.array([ground_pos[0], ground_pos[1], 0.0])
    head_w = np.array([ground_pos[0], ground_pos[1], true_height_m])
    foot_px = project(foot_w)
    head_px = project(head_w)

    est_cm = estimate_height_cm(foot_px, head_px, calib)
    assert abs(est_cm - true_height_m * 100.0) < 0.5  # within 5mm on noiseless synthetic data


def test_ground_position_recovery(tmp_path):
    calib_path, project = make_synthetic_calibration(tmp_path)
    calib = CameraCalibration(calib_path)

    true_pos = (1.5, 9.0)
    foot_px = project([true_pos[0], true_pos[1], 0.0])
    x, y = ground_position_m(foot_px, calib)
    assert abs(x - true_pos[0]) < 0.01
    assert abs(y - true_pos[1]) < 0.01


def test_different_camera_geometry(tmp_path):
    """Sanity check the method isn't tuned to one specific pose."""
    calib_path, project = make_synthetic_calibration(tmp_path, cam_height=6.0, tilt_deg=35.0, fx=1400, fy=1400)
    calib = CameraCalibration(calib_path)

    foot_px = project([0.8, 11.0, 0.0])
    head_px = project([0.8, 11.0, 1.82])
    est_cm = estimate_height_cm(foot_px, head_px, calib)
    assert abs(est_cm - 182.0) < 0.5
