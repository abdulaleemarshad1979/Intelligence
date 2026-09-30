"""City-Wide 600-Camera Topology & Calibration Generator.

Generates and manages a high-density surveillance topology for up to 600 cameras
across municipal districts, arterial transit corridors, and high-crowd pilgrimage zones
(Godavari Pushkaralu Ghats, Kakinada Smart City, Rajahmundry Hubs).
"""

import os
import yaml
import math
import random
from typing import Dict, Any, List, Optional

ZONES_AND_STATIONS = [
    {
        "subdivision": "East Zone",
        "police_station": "II Town Police Station",
        "base_lat": 16.9890,
        "base_lon": 82.2475,
        "landmarks": [
            "Government General Hospital", "Medical College Gate", "Sub-Jail Road",
            "Beach Road Promenade", "Port Terminal Gate", "Suryaraopeta Junction",
            "Jagannaickpur Bridge", "Daba Gardens Crossing", "Kakinada Port Railway"
        ]
    },
    {
        "subdivision": "Central Division",
        "police_station": "I Town Police Station",
        "base_lat": 16.9950,
        "base_lon": 82.2420,
        "landmarks": [
            "District Collectorate", "Municipal Corporation HQ", "RTC Central Bus Complex",
            "Main Market Bazaar", "Clock Tower Square", "Cinema Road Junction",
            "Bhanugudi Junction", "Feeder Road Plaza", "Old Bus Stand Checkpoint"
        ]
    },
    {
        "subdivision": "North Division",
        "police_station": "North Sub-Station",
        "base_lat": 17.0050,
        "base_lon": 82.2460,
        "landmarks": [
            "Ring Road North Terminal", "Smart City Command Hub", "District Police Office",
            "JNTU Engineering Gate", "Nagaram Checkpoint", "Industrial Estate Road",
            "Atchampeta Flyover", "Outer Ring Junction", "Sarpavaram Junction"
        ]
    },
    {
        "subdivision": "South Division",
        "police_station": "South Port PS",
        "base_lat": 16.9720,
        "base_lon": 82.2380,
        "landmarks": [
            "Deep Water Port Gate", "Fertilizer City Feeder", "Coromandel Junction",
            "Beach Bypass Checkpost", "Fisheries Harbor", "Naval Coastal Battery",
            "Kakinada Canal Lock", "South Transit Avenue", "Port Highway Toll"
        ]
    },
    {
        "subdivision": "Pushkaralu Ghats Zone",
        "police_station": "Riverfront Police Post",
        "base_lat": 17.0000,
        "base_lon": 81.7780,
        "landmarks": [
            "VIP Ghat North Gate", "Pushkar Main Ghat Corridor", "Saraswati Ghat Stairs",
            "Kotilingala Temple Entry", "Ramakrishna Mission Ghat", "Feeder Line 1 Ingress",
            "Feeder Line 2 Egress", "Riverfront Holding Area", "Godavari Rail Bridge Approach"
        ]
    },
    {
        "subdivision": "Anaparthi & Transit Circle",
        "police_station": "Anaparthi PS",
        "base_lat": 16.8700,
        "base_lon": 81.9480,
        "landmarks": [
            "Main Road Bus Shelter", "Canal Bridge Checkpoint", "Railway Station East Gate",
            "Market Yard Entrance", "State Highway 40 Checkpost", "Biccavolu Feeder Crossing",
            "Canal Bund Perimeter", "Rice Mill Cluster Road", "Transit Avenue South"
        ]
    }
]


def generate_600_camera_topology(
    base_yaml_path: Optional[str] = None,
    target_count: int = 600,
    seed: int = 42
) -> Dict[str, Any]:
    """Build a calibrated 600-camera registry, preserving existing reference cameras."""
    random.seed(seed)
    cameras: Dict[str, Any] = {}

    # 1. Load and preserve base reference cameras if available
    if base_yaml_path and os.path.isfile(base_yaml_path):
        try:
            with open(base_yaml_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
                cameras = raw.get("cameras", {})
        except Exception:
            pass

    existing_count = len(cameras)
    total_needed = max(target_count, existing_count)

    # Pre-index existing camera IDs
    existing_ids = set(cameras.keys())

    # Build sequence of camera IDs up to target_count (e.g. CAM-001 through CAM-600)
    needed_ids: List[str] = []
    for i in range(1, total_needed + 1):
        cid = f"CAM-{i:03d}"
        if cid not in existing_ids:
            needed_ids.append(cid)

    # Generate metadata for remaining cameras
    num_zones = len(ZONES_AND_STATIONS)
    for idx, cid in enumerate(needed_ids):
        zone_info = ZONES_AND_STATIONS[idx % num_zones]
        landmark = zone_info["landmarks"][idx % len(zone_info["landmarks"])]
        sector_num = (idx // num_zones) + 1

        # Calculate small spatial jitter along city corridors (approx 50m - 200m steps)
        lat_offset = ((idx // 6) * 0.0012) + ((idx % 3) * 0.0004) - 0.002
        lon_offset = ((idx // 6) * 0.0015) + ((idx % 2) * 0.0005) - 0.002
        lat = round(zone_info["base_lat"] + lat_offset, 6)
        lon = round(zone_info["base_lon"] + lon_offset, 6)

        # Standard municipal PTZ / Fixed camera parameters
        is_hd = (idx % 3 != 0)
        res = [1920, 1080] if is_hd else [1280, 720]
        mounting_h = round(4.0 + (idx % 5) * 0.35, 1)  # 4.0m to 5.4m
        tilt = round(32.0 + (idx % 6) * 2.0, 1)        # 32° to 42°
        focal_px = round(950.0 + (idx % 5) * 80.0, 1)
        ground_y = 980 if is_hd else 650

        # Construct realistic camera entity
        cam_num = int(cid.split("-")[-1])
        adj_candidates = []
        if cam_num > 1:
            adj_candidates.append(f"CAM-{cam_num - 1:03d}")
        if cam_num < total_needed:
            adj_candidates.append(f"CAM-{cam_num + 1:03d}")
        # Add cross-corridor link for rich topology graph
        cross_link = f"CAM-{(cam_num + 7) % total_needed + 1:03d}"
        if cross_link != cid and cross_link not in adj_candidates:
            adj_candidates.append(cross_link)

        cameras[cid] = {
            "name": f"{landmark} Sec-{sector_num}",
            "location": f"{zone_info['subdivision']} - {landmark}",
            "subdivision": zone_info["subdivision"],
            "police_station": zone_info["police_station"],
            "latitude": lat,
            "longitude": lon,
            "resolution": res,
            "mounting_height_m": mounting_h,
            "tilt_angle_deg": tilt,
            "focal_length_px": focal_px,
            "ground_plane_y": ground_y,
            "adjacent_cameras": adj_candidates[:3]
        }

    return cameras


def export_600_camera_yaml(output_path: str, base_yaml_path: Optional[str] = None) -> str:
    """Generate and write the full 600-camera configuration to YAML."""
    cams = generate_600_camera_topology(base_yaml_path=base_yaml_path, target_count=600)
    data = {
        "metadata": {
            "title": "High-Density Municipal CCTV Topology (600 Cameras)",
            "total_cameras": len(cams),
            "sectors": [z["subdivision"] for z in ZONES_AND_STATIONS],
            "calibrated_resolution_default": [1920, 1080],
            "version": "2.0.0-scale"
        },
        "cameras": cams
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)
    return output_path
