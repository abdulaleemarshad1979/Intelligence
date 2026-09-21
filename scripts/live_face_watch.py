#!/usr/bin/env python3
"""
live_face_watch.py
--------------------------------------------------------------------
Live CCTV Feed Face Surveillance & Automatic Snapshot Capture Tool.

Given a target face image, continuously monitors a live RTSP camera feed or
surveillance video stream. When the target face appears in the feed, it
automatically captures and stores:
  1. Full scene high-resolution frame snapshot
  2. Isolated face crop snapshot
  3. Evidentiary SHA-256 integrity hash for BSA Section 63 compliance

USAGE
    python scripts/live_face_watch.py \
        --target-face data/targets/suspect.jpg \
        --name "Suspect Raju" \
        --stream rtsp://192.168.1.100:554/media/video1 \
        --camera CAM-001 \
        --threshold 0.55 \
        --out-dir data/captures

    For offline test with video sample:
    python scripts/live_face_watch.py \
        --target-face data/targets/suspect.jpg \
        --name "Suspect Raju" \
        --stream data/samples/cctv_sample_clean.mp4 \
        --camera CAM-001 \
        --threshold 0.50
"""

import argparse
import os
import sys
import time
import logging
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vision.face_watch import LiveFaceWatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_face_watch")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-face", required=True, help="Path to reference face image of the target")
    parser.add_argument("--name", default="Unknown Target", help="Name / designation of the target person")
    parser.add_argument("--stream", default="data/samples/cctv_sample_clean.mp4", help="RTSP URL, video file, or camera device index")
    parser.add_argument("--camera", default="CAM-001", help="Camera ID (e.g. CAM-001)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Biometric match cosine similarity threshold (0.0 to 1.0)")
    parser.add_argument("--out-dir", default="data/captures", help="Directory where captured snapshot images will be stored")
    parser.add_argument("--cooldown", type=float, default=4.0, help="Cooldown in seconds between captures of the same target")
    parser.add_argument("--max-captures", type=int, default=0, help="Stop after N captures (0 = run indefinitely)")
    parser.add_argument("--display", action="store_true", help="Show live visual window with bounding boxes")
    args = parser.parse_args()

    if not os.path.isfile(args.target_face):
        sys.exit(f"Error: Target face image not found at '{args.target_face}'")

    watcher = LiveFaceWatcher(
        storage_dir=args.out_dir,
        cooldown_sec=args.cooldown,
        default_threshold=args.threshold
    )

    print(f"\n=================================================================")
    print(f"       POLICE COMMAND CENTER - LIVE FACE SURVEILLANCE           ")
    print(f"=================================================================")
    print(f"Target Name     : {args.name}")
    print(f"Reference Image : {args.target_face}")
    print(f"CCTV Stream     : {args.stream}")
    print(f"Camera ID       : {args.camera}")
    print(f"Match Threshold : {args.threshold:.2f}")
    print(f"Capture Output  : {args.out_dir}")
    print(f"=================================================================\n")

    # Enroll target face
    try:
        enrolled = watcher.enroll_target_face(
            image_input=args.target_face,
            name=args.name,
            threshold=args.threshold
        )
        print(f"[+] Successfully enrolled target '{args.name}' (ID: {enrolled['target_id']})")
    except Exception as e:
        sys.exit(f"[-] Failed to enroll target face: {e}")

    # Open video capture
    source = int(args.stream) if args.stream.isdigit() else args.stream
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[-] Could not open camera stream/video source: {args.stream}")

    total_frames = 0
    total_captures = 0
    t0 = time.time()

    print("[*] Starting live CCTV feed monitoring... Press Ctrl+C or 'q' to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # If reading a video file and reached end, loop or exit
                if not str(source).startswith("rtsp://") and not str(source).isdigit():
                    print("[*] Reached end of video file.")
                    break
                time.sleep(0.05)
                continue

            total_frames += 1

            # Process live frame against enrolled target faces
            matches = watcher.process_frame(
                frame=frame,
                camera_id=args.camera,
                frame_id=total_frames
            )

            for match in matches:
                total_captures += 1
                print("\n" + "!" * 70)
                print(f" [!!! LIVE TARGET MATCH DETECTED !!!]")
                print(f" Target           : {match['target_name']} (ID: {match['target_id']})")
                print(f" Camera ID        : {match['camera_id']} | Frame: #{match['frame_id']}")
                print(f" Match Confidence : {match['similarity_pct']}% (Score: {match['confidence']:.4f})")
                print(f" Timestamp        : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(match['timestamp']))}")
                print(f" Full Scene Image : {match['full_frame_path']}")
                print(f" Face Crop Image  : {match['face_crop_path']}")
                print(f" SHA-256 Hash     : {match['raw_frame_hash']}")
                print("!" * 70 + "\n")

                if args.max_captures > 0 and total_captures >= args.max_captures:
                    print(f"[*] Reached target maximum captures ({args.max_captures}).")
                    return

            # Optional visual display
            if args.display:
                display_frame = frame.copy()
                for match in matches:
                    fx, fy, fw, fh = match["face_bbox"]
                    cv2.rectangle(display_frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)
                    cv2.putText(
                        display_frame,
                        f"TARGET: {match['target_name']} ({match['similarity_pct']}%)",
                        (fx, max(20, fy - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )
                cv2.imshow(f"Face Surveillance - {args.camera}", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n[*] Surveillance interrupted by operator.")
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()

        elapsed = max(0.1, time.time() - t0)
        fps = total_frames / elapsed
        print(f"\n[*] Session Summary:")
        print(f"    Frames processed : {total_frames} ({fps:.1f} FPS)")
        print(f"    Matches captured : {total_captures}")
        print(f"    Captures dir     : {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
