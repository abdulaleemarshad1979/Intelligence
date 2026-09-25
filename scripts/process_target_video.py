#!/usr/bin/env python3
"""CLI tool to extract exo-skeleton keypoints, gait kinematics, enhance video, and save to SQL for model training.

Usage:
    # Process sample video and extract primary subject
    python scripts/process_target_video.py --video data/samples/cctv_sample_clean.mp4 --name "Primary Subject"

    # Specify a target bounding box at frame 10
    python scripts/process_target_video.py --video data/samples/cctv_sample_clean.mp4 --box 200,80,380,450 --frame 10 --name "Target Alpha"

    # Export 100 frames with video enhancement into SQL and training dataset
    python scripts/process_target_video.py --video /path/to/video.mp4 --max-frames 100
"""

import os
import sys
import time
import argparse
import json

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.pipeline.video_target_processor import VideoTargetProcessor


def main():
    parser = argparse.ArgumentParser(description="Extract exo-skeleton, gait kinematics, enhance video, and save to SQL.")
    parser.add_argument("--video", type=str, default="data/samples/cctv_sample_clean.mp4",
                        help="Path to surveillance video file")
    parser.add_argument("--name", type=str, default="Subject of Interest",
                        help="Name or alias of the target person")
    parser.add_argument("--id", type=str, default=None,
                        help="Target ID (e.g. POI-001)")
    parser.add_argument("--box", type=str, default=None,
                        help="Bounding box hint x1,y1,x2,y2 (e.g. 100,50,250,380)")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame index for bounding box hint")
    parser.add_argument("--track-id", type=str, default=None,
                        help="Target track ID if already known")
    parser.add_argument("--camera-id", type=str, default="CAM-001",
                        help="Surveillance camera ID")
    parser.add_argument("--max-frames", type=int, default=150,
                        help="Maximum frames to process (None for entire video)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Frame processing stride (1 = every frame)")
    parser.add_argument("--no-enhance", action="store_true",
                        help="Disable video enhancement")
    parser.add_argument("--output-dir", type=str, default="data/captures",
                        help="Output directory for enhanced video and training artifacts")
    parser.add_argument("--conf-threshold", type=float, default=0.30,
                        help="Keypoint confidence gating threshold")

    args = parser.parse_args()

    box = None
    if args.box:
        try:
            box = [int(v.strip()) for v in args.box.split(",")]
            assert len(box) == 4
        except Exception:
            print(f"[!] Error: Invalid --box format. Use x1,y1,x2,y2 (e.g. 100,50,250,380)")
            sys.exit(1)

    print("=" * 80)
    print(" CCTV INTELLIGENCE — TARGET EXO-SKELETON & GAIT EXTRACTOR")
    print("=" * 80)
    print(f"[*] Input Video:    {args.video}")
    print(f"[*] Target Person:  {args.name} (ID: {args.id or 'AUTO'})")
    if box:
        print(f"[*] Target Box:     {box} at frame {args.frame}")
    print(f"[*] Max Frames:     {args.max_frames} (Stride: {args.stride})")
    print(f"[*] Enhancement:    {'Enabled (Tier-1 CLAHE + Bilateral)' if not args.no_enhance else 'Disabled'}")
    print(f"[*] Output Dir:     {args.output_dir}")
    print("-" * 80)

    processor = VideoTargetProcessor(
        conf_threshold=args.conf_threshold,
        enable_clahe=not args.no_enhance,
        enable_denoise=not args.no_enhance
    )

    t0 = time.time()
    try:
        results = processor.process_video_target(
            video_path=args.video,
            target_name=args.name,
            target_id=args.id,
            initial_box=box,
            initial_frame=args.frame,
            target_track_id=args.track_id,
            camera_id=args.camera_id,
            max_frames=args.max_frames,
            stride_step=args.stride,
            enhance_video=not args.no_enhance,
            output_dir=args.output_dir
        )
    except Exception as ex:
        print(f"[!] Processing Error: {ex}")
        sys.exit(1)

    elapsed = time.time() - t0
    target = results["target"]
    skel = results["exo_skeleton"]
    gait = results["gait_kinematics"]
    enh = results["video_enhancement"]
    bio = results["biometrics"]
    train = results["model_training_dataset"]

    print("\n" + "=" * 80)
    print(" PROCESSING COMPLETE — SUMMARY REPORT")
    print("=" * 80)
    print(f"[*] Target Locked:         {target['name']} (ID: {target['target_id']}, Track: {target['track_id']})")
    print(f"[*] Best Snapshot Crop:    {target['best_crop_path']}")
    print(f"    SHA-256 Hash:          {target['best_crop_sha256'][:16]}...")
    print(f"[*] Enhanced Video File:   {enh.get('enhanced_video_path', 'N/A')}")
    print(f"    Raw Video SHA-256:     {enh.get('raw_sha256', '')[:16]}...")
    print(f"    Enhanced SHA-256:      {enh.get('enhanced_sha256', '')[:16]}...")
    print(f"[*] Exo-Skeleton Frames:   {skel['frames_extracted']} frames extracted via {skel['model']}")
    print(f"    Saved in SQL Table:    {skel['saved_in_sql']} records into 'person_skeletons'")
    print(f"    Average Confidence:    {skel['average_confidence']:.3f} (gated at >={args.conf_threshold})")
    print("-" * 80)
    print(" GAIT KINEMATICS DYNAMICS (Zero Made-Up Numbers)")
    print("-" * 80)
    print(f"[*] Stride Length:         {gait.get('stride_length_cm', 0.0)} cm ({gait.get('stride_length_px', 0.0)} px)")
    print(f"[*] Cadence:               {gait.get('cadence_steps_per_sec', 0.0)} steps/sec")
    print(f"[*] Spine Tilt Lean:       {gait.get('spine_tilt_deg', 0.0)} degrees")
    print(f"[*] Posture Correctness:   {gait.get('posture_correctness', 0.0)}")
    print(f"[*] Mean Knee Velocity:    {gait.get('mean_knee_angular_velocity_deg_s', 0.0)} deg/s")
    print(f"[*] Mean Hip Velocity:     {gait.get('mean_hip_angular_velocity_deg_s', 0.0)} deg/s")
    print(f"[*] FFT Harmonic Ratios:   {gait.get('fft_harmonic_ratios', [])}")
    print(f"[*] Gait Signal Valid:     {gait.get('valid_gait', False)} (Usable for evidence: {gait.get('gait_usable', False)})")
    print("-" * 80)
    print(" MODEL TRAINING ARTIFACTS")
    print("-" * 80)
    print(f"[*] Tensor Shape:          {train['tensor_shape']} [T frames, V=17 joints, C=3 (x,y,conf)]")
    print(f"[*] NumPy Training File:   {train['npy_path']}")
    print(f"[*] Metadata Label File:   {train['metadata_path']}")
    print(f"[*] Compatible GNN Models: {', '.join(train['compatible_models'])}")
    print("=" * 80)
    print(f"[*] Total Time:            {elapsed:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()
