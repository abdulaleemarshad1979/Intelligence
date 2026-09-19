#!/usr/bin/env python3
"""CLI script to download neural model checkpoints from official GitHub releases.

Usage:
    python scripts/download_models.py --all
    python scripts/download_models.py --status
    python scripts/download_models.py --models yolov8n,face_yunet,face_sface,yolov8n_pose,osnet
"""

import os
import sys
import argparse

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.adapters.downloader import model_downloader, GITHUB_MODEL_CATALOG


def format_size(bytes_val: int) -> str:
    mb = bytes_val / (1024 * 1024)
    return f"{mb:.2f} MB"


def show_status():
    print("=" * 80)
    print(" CCTV INTELLIGENCE — NEURAL MODEL CHECKPOINTS (GITHUB & HUBS)")
    print("=" * 80)
    status = model_downloader.get_status()

    header = f"{'MODEL ID':<15} | {'CATEGORY':<11} | {'STATUS':<12} | {'LOCAL SIZE':<11} | {'SOURCE'}"
    print(header)
    print("-" * 80)

    for key, item in status.items():
        st = "READY" if item["is_downloaded"] else "MISSING"
        sz = f"{item['size_mb']:.1f} MB" if item["is_downloaded"] else f"({item['expected_size_mb']:.1f} MB)"
        tag = "[DOWNLOADED]" if item["is_downloaded"] else "[PENDING]"
        print(f"{key:<15} | {item['category']:<11} | {tag:<12} | {sz:<11} | {item['source']}")

    print("=" * 80)


def download_models(model_keys: list):
    print("=" * 80)
    print(f" DOWNLOADING {len(model_keys)} NEURAL MODELS FROM OFFICIAL GITHUB RELEASES")
    print("=" * 80)

    for key in model_keys:
        if key not in GITHUB_MODEL_CATALOG:
            print(f"[!] Warning: '{key}' not in catalog. Skipping.")
            continue

        info = GITHUB_MODEL_CATALOG[key]
        print(f"\n[*] Target: {info['name']}")
        print(f"    Source URL: {info['url']}")
        print(f"    Destination: models/{info['filename']}")

        def progress(dl, total):
            pct = (dl / total * 100) if total > 0 else 0
            sys.stdout.write(f"\r    Progress: {pct:5.1f}% ({dl / (1024*1024):.1f}MB / {total / (1024*1024):.1f}MB)")
            sys.stdout.flush()

        try:
            res = model_downloader.download_model(key, progress_cb=progress)
            print(f"\n    Result: {res['status']} ({res['size_mb']} MB)")
        except Exception as e:
            print(f"\n    [ERROR] Download failed: {e}")

    print("\n" + "=" * 80)
    print(" Model check complete.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Download CCTV Intelligence model weights from GitHub.")
    parser.add_argument("--all", action="store_true", help="Download all essential models")
    parser.add_argument("--status", action="store_true", help="Show current model status in models/")
    parser.add_argument("--models", type=str, help="Comma-separated model keys to download (e.g. yolov8n,face_yunet)")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.models:
        keys = [k.strip() for k in args.models.split(",") if k.strip()]
        download_models(keys)
        show_status()
        return

    if args.all or len(sys.argv) == 1:
        essential_keys = [k for k, v in GITHUB_MODEL_CATALOG.items() if v.get("is_essential", False)]
        download_models(essential_keys)
        show_status()


if __name__ == "__main__":
    main()
