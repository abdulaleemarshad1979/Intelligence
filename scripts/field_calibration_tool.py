"""
field_calibration_tool.py
--------------------------------------------------------------------
One-time per-camera field calibration for CCTV height/ground-position
estimation.

WHY THIS EXISTS
A single pinhole camera cannot recover metric distances from one frame
unless it knows (a) its own intrinsics (focal length, optical centre)
and (b) its exact pose relative to the ground plane (height above
ground + tilt/roll). This tool solves (b) in the field: a technician
clicks 4+ points on the camera's live frame whose real-world ground
coordinates are known (e.g. painted lane markings, tile joints, a
measuring tape laid on the ground), and the tool solves for the exact
camera pose with cv2.solvePnP. This is the same set of correspondences
you'd otherwise use for a plain ground-plane homography, but solvePnP
also gives you the camera's 3D position and tilt, which height
estimation needs (see height_estimation.py).

INTRINSICS
Camera intrinsics (fx, fy, cx, cy) should be calibrated ONCE per
camera/lens MODEL in a lab (checkerboard calibration, cv2.calibrateCamera)
and reused across every pole using that model, not re-done per install.
Pass them with --fx/--fy/--cx/--cy, or supply --hfov-deg for a rough
estimate from the image width if no lab calibration exists yet
(flag this in your compliance notes as a source of extra error).

USAGE
    python field_calibration_tool.py \
        --image frame_cam014.jpg \
        --camera-id CAM-014 \
        --fx 1400 --fy 1400 \
        --out calibrations/CAM-014.json

    Click at least 4 points on the displayed frame (ground-plane points
    spread across the field of view give the most robust result — avoid
    clustering all points close together). After each click you'll be
    prompted in the terminal for that point's real-world (X, Y) ground
    coordinates in metres, relative to any fixed origin you choose
    (e.g. a marked reference tile). Press 'q' once you've entered all
    points to solve and save the calibration.

OUTPUT
    A JSON file with the camera's intrinsics, solved pose (rvec/tvec),
    camera height/tilt for a quick sanity read, and the mean
    reprojection error in pixels — LOOK AT THIS NUMBER. Under ~2px on a
    1080p frame is a solid calibration; above ~5px, re-do it (a
    mis-clicked point or a wrong real-world coordinate is the usual
    cause).
"""
import argparse
import json
import sys
from datetime import datetime, timezone

import cv2
import numpy as np

img_points = []
world_points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        img_points.append([float(x), float(y)])
        print(f"\nClicked image point #{len(img_points)}: ({x}, {y})")
        try:
            wx = float(input("  real-world X (metres): "))
            wy = float(input("  real-world Y / depth (metres): "))
        except ValueError:
            print("  Invalid number, discarding this point.")
            img_points.pop()
            return
        world_points.append([wx, wy, 0.0])  # ground plane: Z = 0
        display = param["display"]
        cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(display, str(len(img_points)), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def build_intrinsics(args, img_w, img_h):
    if args.fx and args.fy:
        fx, fy = args.fx, args.fy
    elif args.hfov_deg:
        fx = fy = (img_w / 2.0) / np.tan(np.deg2rad(args.hfov_deg) / 2.0)
        print(f"[warn] Using approximate focal length from HFOV: fx=fy={fx:.1f}px. "
              f"Do a proper lab checkerboard calibration for this camera model when possible.")
    else:
        sys.exit("Provide either --fx/--fy (preferred) or --hfov-deg.")
    cx = args.cx if args.cx else img_w / 2.0
    cy = args.cy if args.cy else img_h / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="Path to a still frame from the camera to calibrate")
    ap.add_argument("--camera-id", required=True)
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--hfov-deg", type=float, default=None,
                     help="Fallback if fx/fy unknown: horizontal field of view in degrees")
    ap.add_argument("--out", default=None, help="Output JSON path (default: <camera-id>.json)")
    args = ap.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        sys.exit(f"Could not read image: {args.image}")
    h, w = frame.shape[:2]
    K = build_intrinsics(args, w, h)

    display = frame.copy()
    cv2.namedWindow("calibration - click ground points, then press q")
    cv2.setMouseCallback("calibration - click ground points, then press q", mouse_callback, {"display": display})

    print("Click ground-plane reference points (min 4, spread across the frame).")
    print("Press 'q' in the image window when done.\n")
    while True:
        cv2.imshow("calibration - click ground points, then press q", display)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

    if len(img_points) < 4:
        sys.exit(f"Need at least 4 points, got {len(img_points)}. Re-run.")

    img_pts = np.array(img_points, dtype=np.float64)
    world_pts = np.array(world_points, dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(world_pts, img_pts, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        sys.exit("solvePnP failed to converge — check your clicked points and coordinates.")

    # Reprojection error: the single most important QA number for this calibration.
    reproj, _ = cv2.projectPoints(world_pts, rvec, tvec, K, None)
    reproj = reproj.reshape(-1, 2)
    errors = np.linalg.norm(reproj - img_pts, axis=1)
    mean_err = float(np.mean(errors))

    R, _ = cv2.Rodrigues(rvec)
    cam_center = (-R.T @ tvec).flatten()
    # Approx tilt below horizontal, for a human-readable sanity check
    forward_world = R.T @ np.array([0, 0, 1.0])
    tilt_deg = float(np.degrees(np.arctan2(-forward_world[2], np.linalg.norm(forward_world[:2]))))

    calib = {
        "camera_id": args.camera_id,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "image_size": [w, h],
        "K": K.tolist(),
        "rvec": rvec.flatten().tolist(),
        "tvec": tvec.flatten().tolist(),
        "camera_center_world_m": cam_center.tolist(),
        "camera_height_m": float(cam_center[2]),
        "approx_tilt_deg_below_horizontal": tilt_deg,
        "n_calibration_points": len(img_points),
        "mean_reprojection_error_px": mean_err,
        "max_reprojection_error_px": float(np.max(errors)),
    }

    out_path = args.out or f"{args.camera_id}.json"
    with open(out_path, "w") as f:
        json.dump(calib, f, indent=2)

    print(f"\nSaved calibration to {out_path}")
    print(f"Camera height: {calib['camera_height_m']:.2f} m, tilt: {tilt_deg:.1f} deg")
    print(f"Mean reprojection error: {mean_err:.2f} px "
          f"({'OK' if mean_err < 2 else 'HIGH — recalibrate, a point is likely off'})")


if __name__ == "__main__":
    main()
