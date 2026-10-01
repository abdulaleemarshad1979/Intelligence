#!/usr/bin/env python3
"""Main Application Launcher for City-Wide Intelligent CCTV Intelligence Platform.

Usage:
    export ICSEE_CAMERA_IP=10.243.1.65
    export ICSEE_CAMERA_USER=rtsp
    export ICSEE_CAMERA_PASSWORD='Test1234'
    python3 app.py

Optional multi-camera / fleet environment options:
    export ICSEE_CAMERA_ID=CAM-001              # Target slot (default CAM-001)
    export ICSEE_CAMERA_PORT=554                # RTSP port (default 554)
    export ICSEE_STREAM=stream0                 # Profile: stream0 (1080p), stream1 (sub), onvif1
    export ICSEE_CAMERA_COUNT=10                # Auto-connect sequential IP range (10.243.1.65..74)
    export ICSEE_CAMERA_IPS="10.243.1.65,10.243.1.66" # Comma-separated list of camera IPs
"""

import os
import sys

# Seamlessly fallback to project .venv if executed with system Python
_venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python3")
if os.path.exists(_venv_py) and os.path.abspath(sys.executable) != os.path.abspath(_venv_py):
    try:
        import uvicorn
        import fastapi
    except ImportError:
        os.execv(_venv_py, [_venv_py] + sys.argv)

import ipaddress
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Allow module to act as package facade so `from app.main import ...` works seamlessly
__path__ = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")]

# Configure base logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("CCTV-Platform")

def setup_icsee_camera():
    """Verify and initialize ICSee IP Camera connection if environment variables are set."""
    icsee_ip = os.getenv("ICSEE_CAMERA_IP", "").strip().strip("'\"")
    icsee_ips_raw = os.getenv("ICSEE_CAMERA_IPS", "").strip().strip("'\"")
    icsee_cameras_raw = os.getenv("ICSEE_CAMERAS", "").strip().strip("'\"")

    if not icsee_ip and not icsee_ips_raw and not icsee_cameras_raw:
        logger.info("No ICSEE_CAMERA_IP environment variable set. Running all 600 cameras in fleet-ready standby/simulation mode.")
        return

    icsee_user = os.getenv("ICSEE_CAMERA_USER", "rtsp").strip().strip("'\"")
    icsee_pass = os.getenv("ICSEE_CAMERA_PASSWORD", "").strip().strip("'\"")
    icsee_port = int(os.getenv("ICSEE_CAMERA_PORT", "554"))
    icsee_stream = os.getenv("ICSEE_STREAM", "stream0").strip().strip("'\"")
    base_cam_id = os.getenv("ICSEE_CAMERA_ID", "CAM-001").strip().upper().strip("'\"")
    try:
        count = int(os.getenv("ICSEE_CAMERA_COUNT", "1"))
    except Exception:
        count = 1

    from app.ingestion.stream_manager import build_icsee_rtsp_url, get_stream_manager
    from app.main import CCTV_CAMERAS_REGISTRY, camera_configs, repo
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    stream_mgr = get_stream_manager(data_dir)

    cameras_to_connect = []

    # 1. Check explicit mapping format (e.g. CAM-001=10.243.1.65,CAM-002=10.243.1.66)
    if icsee_cameras_raw:
        for item in icsee_cameras_raw.split(","):
            if "=" in item:
                slot, ip = item.split("=", 1)
                cameras_to_connect.append((slot.strip().upper(), ip.strip()))
    # 2. Check comma-separated IPs (e.g. 10.243.1.65,10.243.1.66)
    elif icsee_ips_raw:
        try:
            start_num = int(base_cam_id.replace("CAM-", "")) if "CAM-" in base_cam_id else 1
        except Exception:
            start_num = 1
        for idx, ip_str in enumerate(icsee_ips_raw.split(",")):
            c_num = start_num + idx
            cameras_to_connect.append((f"CAM-{c_num:03d}", ip_str.strip()))
    # 3. Single IP or sequential range count
    elif icsee_ip:
        try:
            start_num = int(base_cam_id.replace("CAM-", "")) if "CAM-" in base_cam_id else 1
        except Exception:
            start_num = 1
        try:
            base_ip_obj = ipaddress.ip_address(icsee_ip)
            for i in range(max(1, count)):
                c_num = start_num + i
                cameras_to_connect.append((f"CAM-{c_num:03d}", str(base_ip_obj + i)))
        except Exception:
            cameras_to_connect.append((base_cam_id, icsee_ip))

    print("\n" + "=" * 70)
    print("  ANDHRA PRADESH POLICE DEPARTMENT - CCTV SURVEILLANCE PLATFORM")
    print(f"  Fleet Capacity: 600 Streams Ready (Real-time Ingestion & AI Perception)")
    print(f"  Configured Live ICSee Cameras: {len(cameras_to_connect)}")
    print("=" * 70)

    for cid, ip in cameras_to_connect:
        rtsp_url = build_icsee_rtsp_url(
            ip=ip,
            port=icsee_port,
            username=icsee_user,
            password=icsee_pass,
            stream_type=icsee_stream
        )
        cam_name = f"ICSee IP Camera ({ip})"
        cam_loc = f"ICSee Feed ({ip})"
        masked_url = rtsp_url.replace(icsee_pass, '******') if icsee_pass else rtsp_url

        try:
            worker = stream_mgr.attach_icsee_camera(
                camera_id=cid,
                ip=ip,
                port=icsee_port,
                username=icsee_user,
                password=icsee_pass,
                stream_type=icsee_stream,
                name=cam_name
            )
            # Update in-memory registry
            for c in CCTV_CAMERAS_REGISTRY:
                if c["camera_id"] == cid:
                    c["name"] = cam_name
                    c["location"] = cam_loc
                    c["rtmp"] = rtsp_url
                    c["status"] = "ACTIVE"
                    c["ip_address"] = ip
                    break

            if cid in camera_configs:
                camera_configs[cid]["name"] = cam_name
                camera_configs[cid]["location"] = cam_loc
                camera_configs[cid]["rtsp_url"] = rtsp_url

            logger.info(f"Attached live ICSee camera on [{cid}]: {masked_url}")
        except Exception as ex:
            logger.error(f"Error attaching ICSee feed on [{cid}]: {ex}", exc_info=True)

    print("=" * 70 + "\n")


def main():
    """Boot application server with 600 streams capacity and ICSee integration."""
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python3")
    if os.path.isfile(venv_python) and sys.executable != venv_python:
        try:
            import uvicorn
        except ImportError:
            os.execv(venv_python, [venv_python] + sys.argv)

    import uvicorn
    setup_icsee_camera()

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() in ("true", "1", "yes")

    logger.info(f"Starting CCTV Intelligence Command Center on http://{host}:{port} (Capacity: 600 Cameras)")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
