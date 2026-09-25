#!/usr/bin/env python3
"""Test Face Search & Match Script against CCTV Sample Video.

Usage:
    python scripts/test_face_match_on_sample.py --image path/to/target_face.jpg
    python scripts/test_face_match_on_sample.py --image path/to/target_face.jpg --video data/samples/cctv_sample.mp4 --threshold 0.50
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np

# Ensure root workspace directory is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.vision.face_engine import FaceBiometricEngine


def search_face_in_video(image_path: str, video_path: str, threshold: float = 0.50, stride: int = 5, output_dir: str = "data/search_results"):
    """Searches for target_image face inside video_path and saves match frames."""
    if not os.path.isfile(image_path):
        print(f"[!] Error: Target image file not found: {image_path}")
        return False, []

    if not os.path.isfile(video_path):
        print(f"[!] Error: Video file not found: {video_path}")
        return False, []

    engine = FaceBiometricEngine()

    print(f"[*] Target Image: {image_path}")
    print(f"[*] Sample Video: {video_path}")
    print(f"[*] Similarity Threshold: {threshold * 100:.1f}%")
    print(f"[*] Frame Stride: {stride}")
    print("-" * 60)

    # 1. Load target image & extract probe face embedding
    probe_img = cv2.imread(image_path)
    if probe_img is None:
        print(f"[!] Error: Unable to read image: {image_path}")
        return False, []

    probe_faces = engine.detect_faces(probe_img)
    probe_emb = None

    if probe_faces:
        p_box = [int(v) for v in probe_faces[0]['bbox']]
        p_crop = probe_img[max(0, p_box[1]):min(probe_img.shape[0], p_box[1]+p_box[3]),
                           max(0, p_box[0]):min(probe_img.shape[1], p_box[0]+p_box[2])]
        viable, probe_emb = engine.extract_face_embedding(p_crop, landmarks=probe_faces[0].get('landmarks'), full_frame=probe_img)

    if probe_emb is None or not np.any(probe_emb):
        viable, probe_emb = engine.extract_face_embedding(probe_img)

    if probe_emb is None or not np.any(probe_emb) or np.linalg.norm(probe_emb) < 1e-5:
        print("[!] Error: No face could be detected/extracted from target image!")
        return False, []

    probe_norm = probe_emb / np.linalg.norm(probe_emb)
    print(f"[✓] Successfully extracted probe face embedding (norm={np.linalg.norm(probe_norm):.2f})")

    # 2. Scan Video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_sec = total_frames / fps

    print(f"[*] Video Details: {total_frames} total frames, {fps:.1f} FPS ({duration_sec:.1f}s duration)")
    print("[*] Scanning video frames for target face matches...\n")

    os.makedirs(output_dir, exist_ok=True)
    matches = []

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            timestamp_sec = frame_idx / fps
            faces = engine.detect_faces(frame)

            for f_idx, f in enumerate(faces):
                bx, by, bw, bh = [int(v) for v in f['bbox']]
                bx1, by1 = max(0, bx), max(0, by)
                bx2, by2 = min(frame.shape[1], bx + bw), min(frame.shape[0], by + bh)

                if (bx2 - bx1) >= 20 and (by2 - by1) >= 20:
                    crop = frame[by1:by2, bx1:bx2]
                    viable, emb = engine.extract_face_embedding(crop, landmarks=f.get('landmarks'), full_frame=frame)

                    if np.any(emb) and np.linalg.norm(emb) > 1e-5:
                        cand_norm = emb / np.linalg.norm(emb)
                        sim = float(np.dot(probe_norm, cand_norm))

                        if sim >= threshold:
                            match_info = {
                                "frame_idx": frame_idx,
                                "timestamp_sec": round(timestamp_sec, 2),
                                "similarity": round(sim, 4),
                                "similarity_pct": round(sim * 100, 1),
                                "bbox": [bx, by, bw, bh]
                            }

                            # Save visual verification snapshot
                            vis_frame = frame.copy()
                            cv2.rectangle(vis_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                            cv2.putText(
                                vis_frame,
                                f"MATCH: {sim*100:.1f}% ({timestamp_sec:.1f}s)",
                                (bx, max(20, by - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2
                            )
                            snap_path = os.path.join(output_dir, f"match_frame_{frame_idx}_t{int(timestamp_sec)}s_{int(sim*100)}pct.jpg")
                            cv2.imwrite(snap_path, vis_frame)
                            match_info["snapshot_path"] = snap_path
                            matches.append(match_info)

                            print(f"  [✓] MATCH DETECTED! Frame {frame_idx:4d} | Time {timestamp_sec:5.1f}s | Confidence: {sim*100:5.1f}% | Saved: {snap_path}")

        frame_idx += 1

    cap.release()
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print(" FACE SEARCH COMPLETED")
    print("=" * 60)
    print(f"Frames Processed: {frame_idx}")
    print(f"Time Taken:       {elapsed:.2f} seconds ({frame_idx / max(0.01, elapsed):.1f} FPS)")
    print(f"Matches Found:    {len(matches)}")

    if matches:
        matches.sort(key=lambda m: m["similarity"], reverse=True)
        best = matches[0]
        print(f"\n[★] BEST MATCH: Frame {best['frame_idx']} at {best['timestamp_sec']}s with {best['similarity_pct']}% confidence!")
        print(f"[★] Snapshot saved to: {best['snapshot_path']}")

    return True, matches


def main():
    parser = argparse.ArgumentParser(description="Test target face image against CCTV sample video")
    parser.add_argument("--image", type=str, required=True, help="Path to probe/target face image")
    parser.add_argument("--video", type=str, default=os.path.join(ROOT_DIR, "data", "samples", "cctv_sample.mp4"),
                        help="Path to sample video file")
    parser.add_argument("--threshold", type=float, default=0.45, help="Similarity confidence threshold (default: 0.45 = 45 percent)")
    parser.add_argument("--stride", type=int, default=3, help="Frame processing stride (default: 3 = every 3rd frame)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(ROOT_DIR, "data", "search_results"),
                        help="Directory to save match snapshots")
    args = parser.parse_args()

    search_face_in_video(
        image_path=args.image,
        video_path=args.video,
        threshold=args.threshold,
        stride=args.stride,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
