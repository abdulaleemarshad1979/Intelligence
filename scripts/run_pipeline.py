"""End-to-end CCTV Video Processing Pipeline."""

import os
import sys
import cv2
import time
import argparse
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.detection.person_detector import PersonDetector
from app.tracking.tracker import MultiPersonTracker
from app.tracking.track_manager import TrackManager
from app.features.face import FaceAnalyzer
from app.features.body import BodyAnalyzer
from app.features.height import HeightEstimator
from app.features.pose import PoseEstimator
from app.features.gait import GaitAnalyzer
from app.reid.matcher import CandidateMatcher
from app.database.models import TrackObservation
from app.database.repository import Repository
from app.database.database import init_db

def run_pipeline(
    video_path: str = "data/samples/cctv_sample.mp4",
    camera_id: str = "CAM-001",
    max_frames: int = 400,
    stride_step: int = 2,
    output_annotated_video: str = ""
):
    init_db()
    repo = Repository()
    
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Failed to open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[*] Opened CCTV video: {video_path} ({total_video_frames} frames, {fps:.1f} FPS)")

    detector = PersonDetector(min_height=65, min_width=25)
    tracker = MultiPersonTracker(max_age=25, min_hits=2, iou_threshold=0.22)
    track_mgr = TrackManager(storage_dir="data/tracks")
    face_analyzer = FaceAnalyzer()
    body_analyzer = BodyAnalyzer()
    height_estimator = HeightEstimator(mounting_height_m=4.2, tilt_deg=35.0, focal_length_px=950.0)
    pose_estimator = PoseEstimator()
    gait_analyzer = GaitAnalyzer(window_size=24)
    matcher = CandidateMatcher(repository=repo)

    # State per track ID
    pose_buffers = {}
    track_observations = {}

    writer = None
    if output_annotated_video:
        os.makedirs(os.path.dirname(output_annotated_video) or ".", exist_ok=True)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_annotated_video, fourcc, fps / stride_step, (w, h))

    frame_idx = 0
    processed_count = 0
    start_time = time.time()

    print(f"[*] Processing CCTV stream from {camera_id}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % stride_step != 0:
            continue

        processed_count += 1
        if max_frames and processed_count > max_frames:
            break

        h, w = frame.shape[:2]

        # 1. Detection
        detections = detector.detect(frame)

        # 2. Tracking
        active_tracks = tracker.update(detections)

        # 3. Process Each Track
        for trk in active_tracks:
            tid = trk.track_id
            box = trk.box
            x1, y1, x2, y2 = box

            # Crop
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            person_crop = frame[cy1:cy2, cx1:cx2]
            if person_crop.size == 0:
                continue

            # Record frame in TrackManager
            track_mgr.record_frame(tid, camera_id, frame_idx, box, frame)

            # Features: Pose & Stride
            pose_res = pose_estimator.estimate_pose(person_crop, box)
            if tid not in pose_buffers:
                pose_buffers[tid] = []
            pose_buffers[tid].append(pose_res)
            if len(pose_buffers[tid]) > 40:
                pose_buffers[tid].pop(0)

            # Features: Height
            height_res = height_estimator.estimate_height_cm(box, frame_height=h)

            # Features: Face (3-tier partition)
            face_res = face_analyzer.analyze_person_crop(person_crop)

            # Features: Body & Clothing
            body_res = body_analyzer.analyze(person_crop)

            # Features: Gait & Posture
            gait_res = gait_analyzer.analyze_sequence(pose_buffers[tid], estimated_height_cm=height_res["estimated_height_cm"])

            # Maintain best observation
            tb = track_mgr.get_or_create(tid, camera_id)
            best_crop_path = track_mgr.save_best_crop(tid) or ""

            obs = TrackObservation(
                track_id=tid,
                camera_id=camera_id,
                first_seen=tb.first_seen,
                last_seen=tb.last_seen,
                frame_count=len(tb.frame_indices),
                best_frame_path=best_crop_path,
                face_visible=face_res["face_visible"],
                face_status=face_res["face_status"],
                face_tier_details=face_res["tiers"],
                estimated_height_cm=height_res["estimated_height_cm"],
                body_proportions=body_res["proportions"],
                clothing_upper=body_res["clothing"]["upper_hex"],
                clothing_lower=body_res["clothing"]["lower_hex"],
                stride_length_px=gait_res["stride_length_px"],
                stride_length_cm=gait_res["stride_length_cm"],
                cadence_steps_per_sec=gait_res["cadence_steps_per_sec"],
                spine_tilt_deg=gait_res["spine_tilt_deg"],
                posture_score=gait_res["posture_correctness"],
                gait_wave=gait_res["gait_wave"],
                face_embedding=face_res["face_embedding"],
                body_embedding=body_res["body_embedding"],
                gait_embedding=gait_res["gait_embedding"]
            )
            track_observations[tid] = obs
            repo.save_track(obs)

            # Run Evidence Fusion Matching
            if obs.frame_count >= 3 and obs.frame_count % 5 == 0:
                candidates = matcher.match_track(obs, top_k=3)

            # Draw visual overlay on frame if writer is active
            if writer is not None:
                color = (0, 255, 0) if obs.face_visible else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Overlay tag
                status_tag = f"{tid} | H:{obs.estimated_height_cm:.0f}cm | {obs.face_status}"
                cv2.putText(frame, status_tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

                # Draw 3-part face box if detected
                if face_res["is_detected"]:
                    fb = face_res["face_box"]
                    fx1, fy1, fx2, fy2 = x1 + fb[0], y1 + fb[1], x1 + fb[2], y1 + fb[3]
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 255, 0), 1)
                    # Draw tier partition lines
                    th = fy2 - fy1
                    cv2.line(frame, (fx1, int(fy1 + th * 0.33)), (fx2, int(fy1 + th * 0.33)), (255, 200, 0), 1)
                    cv2.line(frame, (fx1, int(fy1 + th * 0.66)), (fx2, int(fy1 + th * 0.66)), (255, 200, 0), 1)

                # Draw skeleton joints
                kp_glob = pose_res.get("keypoints_global", {})
                for jname, (jx, jy) in kp_glob.items():
                    cv2.circle(frame, (jx, jy), 3, (0, 255, 255), -1)

        if writer is not None:
            writer.write(frame)

        if processed_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_proc = processed_count / max(0.001, elapsed)
            print(f"[*] Processed {processed_count} frames | Active Tracks: {len(active_tracks)} | Speed: {fps_proc:.1f} FPS")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[*] Annotated demonstration video saved to: {output_annotated_video}")

    # Final matching run for all tracked subjects
    print("\n" + "="*70)
    print("      MULTI-MODAL CCTV PERSON INTELLIGENCE REPORT")
    print("="*70)
    
    match_events = repo.get_match_events(limit=20)
    print(f"Total Confirmed Tracks Processed: {len(track_observations)}")
    print(f"Total Match Events Logged: {len(match_events)}\n")

    for tid, obs in track_observations.items():
        if obs.frame_count < 3:
            continue
        print(f"--- [TRACK: {tid}] ({obs.frame_count} frames) ---")
        print(f"  Face Status       : {obs.face_status} (Visible: {obs.face_visible})")
        print(f"  Estimated Height  : {obs.estimated_height_cm} cm")
        print(f"  Body Proportions  : Torso/Leg Ratio={obs.body_proportions.get('torso_leg_ratio', 'N/A')}")
        print(f"  Clothing Colors   : Upper={obs.clothing_upper}, Lower={obs.clothing_lower}")
        print(f"  Gait & Posture    : Stride={obs.stride_length_cm}cm, Cadence={obs.cadence_steps_per_sec} steps/s, Posture Score={obs.posture_score}")
        
        # Query top match
        candidates = matcher.match_track(obs, top_k=2)
        if candidates:
            c = candidates[0]
            print(f"  Top Candidate     : {c['suspect_name']} ({c['fir_no']})")
            print(f"  Total Confidence  : {c['total_confidence']*100:.1f}% [{c['status']}]")
            print(f"  Evidence Breakdown: Face={c['scores']['face_score']*100:.1f}%, Body={c['scores']['body_score']*100:.1f}%, Gait={c['scores']['gait_score']*100:.1f}%, Height={c['scores']['height_score']*100:.1f}%")
            print(f"  Recommendation    : {c['recommendation']}")
        print()

    repo.log_audit("PIPELINE_RUN_COMPLETED", f"Processed {processed_count} frames on {camera_id}, identified {len(track_observations)} tracks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CCTV Multi-Modal Intelligence Pipeline")
    parser.add_argument("--video", type=str, default="data/samples/cctv_sample.mp4", help="Path to CCTV video")
    parser.add_argument("--camera", type=str, default="CAM-001", help="Camera ID")
    parser.add_argument("--max-frames", type=int, default=150, help="Maximum frames to process (0 = all)")
    parser.add_argument("--stride", type=int, default=2, help="Frame step stride")
    parser.add_argument("--output-video", type=str, default="", help="Path to save annotated output video")
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        camera_id=args.camera,
        max_frames=args.max_frames,
        stride_step=args.stride,
        output_annotated_video=args.output_video
    )
