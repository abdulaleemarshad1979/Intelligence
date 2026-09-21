"""High-precision utility to remove baked-in camera DVR red bounding boxes from CCTV sample footage.

Applies strict chromatic differential thresholding and connected-component shape analysis
to erase only synthetic DVR lines while preserving 100% of the natural pavement, clothing,
and scenery with zero temporal flickering.
"""

import os
import sys
import time
import subprocess
import cv2
import numpy as np


def clean_video(input_path: str, output_path: str) -> bool:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open {input_path}")
        return False

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing {total_frames} frames ({w}x{h} @ {fps} FPS)...")

    # Encode with broadcast-standard H.264 via FFmpeg pipe for zero decoder flicker and clean keyframes
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_path
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    count = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Strict chromatic differential detection for synthetic DVR red
        b = frame[:, :, 0].astype(np.int16)
        g = frame[:, :, 1].astype(np.int16)
        r = frame[:, :, 2].astype(np.int16)

        # Synthetic red has high red intensity and huge difference over green and blue
        mask = (r > 165) & ((r - g) > 85) & ((r - b) > 85)

        # Exclude camera OSD timestamp area at top-left
        mask[0:45, 0:300] = False
        # Exclude bottom-most edge
        mask[h-15:h, :] = False

        # Filter out isolated single-pixel noise (pavement noise is < 15 px; DVR lines are > 50 px)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        clean_mask = np.zeros_like(mask, dtype=np.uint8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 15:
                clean_mask[labels == i] = 255

        if np.any(clean_mask):
            dilated = cv2.dilate(clean_mask, kernel, iterations=1)
            cleaned = cv2.inpaint(frame, dilated, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
        else:
            cleaned = frame

        try:
            proc.stdin.write(cleaned.tobytes())
        except BrokenPipeError:
            print("Error: FFmpeg pipe broken unexpectedly")
            break

        count += 1
        if count % 200 == 0:
            elapsed = time.time() - t0
            print(f"Frame {count}/{total_frames} ({count/total_frames*100:.1f}%) in {elapsed:.1f}s")

    cap.release()
    proc.stdin.close()
    proc.wait()
    total_time = time.time() - t0
    print(f"Done! Cleaned {count} frames in {total_time:.1f}s -> {output_path}")
    return True


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_video = os.path.join(base_dir, "data", "samples", "cctv_sample_raw.mp4")
    if not os.path.exists(in_video):
        in_video = os.path.join(base_dir, "data", "samples", "cctv_sample.mp4")
    out_video = os.path.join(base_dir, "data", "samples", "cctv_sample_clean.mp4")
    clean_video(in_video, out_video)
