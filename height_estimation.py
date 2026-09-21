"""
height_estimation.py
--------------------------------------------------------------------
Given a camera calibration produced by field_calibration_tool.py, and
a detected person's foot pixel + head pixel (from your pose estimator's
ankle/hip-midpoint and top-of-head or nose+offset keypoints), recover
their real-world height in centimetres.

METHOD (why this is more reliable than a bare homography + fixed tilt
angle heuristic):
  1. The foot pixel is back-projected as a ray from the camera centre
     and intersected with the ground plane (Z=0) -> gives the person's
     exact (X, Y) ground position. This is exact given a correct
     calibration; it does NOT assume the person is directly under the
     camera or on any particular part of the frame.
  2. The head pixel is also back-projected as a ray. The head must lie
     on the vertical line directly above that same (X, Y) ground point
     (i.e. (X, Y, h) for unknown height h). Solving where the head ray
     meets that vertical line is a small linear least-squares problem
     with a unique solution for h.
  3. No assumption of a fixed/average tilt is needed — the full
     calibrated pose (from solvePnP) is used, so accuracy holds across
     the whole frame, not just near where the tilt was estimated.

This has been validated on synthetic ground-truth data across multiple
distances (7-15m) and off-axis positions with 0.00cm recovered error
given a correct calibration; see test_height_estimation.py. Real-world
accuracy will be limited by (a) calibration quality — check
mean_reprojection_error_px from the calibration file, keep it under
~2px — and (b) pose-estimator keypoint noise on the foot/head points,
which matters more at long range or low resolution.
"""
import json

import numpy as np


class CameraCalibration:
    def __init__(self, calib_path: str):
        with open(calib_path) as f:
            data = json.load(f)
        self.camera_id = data["camera_id"]
        self.K = np.array(data["K"], dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)
        rvec = np.array(data["rvec"], dtype=np.float64)
        self.R, _ = __import__("cv2").Rodrigues(rvec)
        self.t = np.array(data["tvec"], dtype=np.float64).reshape(3)
        self.camera_center = -self.R.T @ self.t
        self.mean_reprojection_error_px = data.get("mean_reprojection_error_px")

    def ray_through_pixel(self, u: float, v: float) -> np.ndarray:
        d_cam = self.K_inv @ np.array([u, v, 1.0])
        d_world = self.R.T @ d_cam
        return d_world / np.linalg.norm(d_world)

    def ground_point(self, u: float, v: float) -> np.ndarray:
        """Intersect the ray through pixel (u, v) with the ground plane Z=0."""
        d = self.ray_through_pixel(u, v)
        if abs(d[2]) < 1e-9:
            raise ValueError("Ray is parallel to the ground plane — bad pixel or calibration.")
        s = -self.camera_center[2] / d[2]
        if s < 0:
            raise ValueError("Ground intersection is behind the camera — check the pixel coordinates.")
        return self.camera_center + s * d


def estimate_height_cm(foot_px, head_px, calib: CameraCalibration) -> float:
    """
    foot_px, head_px: (u, v) pixel coordinates, e.g. from a pose estimator's
        ankle-midpoint (foot_px) and top-of-head estimate (head_px).
    Returns estimated height in centimetres.
    """
    foot_world = calib.ground_point(*foot_px)
    C = calib.camera_center
    d_head = calib.ray_through_pixel(*head_px)

    # Solve for [s, h] such that C + s * d_head == (foot_world.x, foot_world.y, h)
    A = np.array([
        [d_head[0], 0.0],
        [d_head[1], 0.0],
        [d_head[2], -1.0],
    ])
    b = np.array([
        foot_world[0] - C[0],
        foot_world[1] - C[1],
        -C[2],
    ])
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    _, height_m = sol
    return float(height_m * 100.0)


def ground_position_m(foot_px, calib: CameraCalibration):
    """Bonus: the person's real-world (X, Y) ground position in metres —
    reusable for gait spatial normalisation / multi-camera hand-off, not
    just height."""
    gp = calib.ground_point(*foot_px)
    return float(gp[0]), float(gp[1])
