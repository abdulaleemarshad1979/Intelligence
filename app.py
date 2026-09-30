#!/usr/bin/env python3
"""Main Application Launcher for City-Wide Intelligent CCTV Intelligence Platform.

Usage:
    export ICSEE_CAMERA_IP=192.168.1.111
    export ICSEE_CAMERA_USER=hmpw
    export ICSEE_CAMERA_PASSWORD='r7h3m2'
    python3 app.py
"""

import os
import sys

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
    icsee_ip = os.getenv("ICSEE_CAMERA_IP", "").strip()
    if not icsee_ip:
        logger.info("No ICSEE_CAMERA_IP environment variable set. Running in default multi-stream simulation mode.")
        return

    icsee_user = os.getenv("ICSEE_CAMERA_USER", "admin").strip()
    icsee_pass = os.getenv("ICSEE_CAMERA_PASSWORD", "").strip()
    icsee_port = int(os.getenv("ICSEE_CAMERA_PORT", "554"))
    icsee_stream = os.getenv("ICSEE_STREAM", "stream0").strip()
    icsee_cam_id = os.getenv("ICSEE_CAMERA_ID", "CAM-001").strip()

    from app.ingestion.stream_manager import build_icsee_rtsp_url, get_stream_manager
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    stream_mgr = get_stream_manager(data_dir)

    rtsp_url = build_icsee_rtsp_url(
        ip=icsee_ip,
        port=icsee_port,
        username=icsee_user,
        password=icsee_pass,
        stream_type=icsee_stream
    )

    print("\n" + "=" * 70)
    print("  ANDHRA PRADESH POLICE DEPARTMENT - CCTV SURVEILLANCE PLATFORM")
    print(f"  Fleet Capacity: 600 Streams Initialized")
    print(f"  Live ICSee Camera Target: {icsee_ip}:{icsee_port}/{icsee_stream}")
    print(f"  User: {icsee_user} | Assigned Slot: {icsee_cam_id}")
    print(f"  RTSP URL: {rtsp_url.replace(icsee_pass, '******') if icsee_pass else rtsp_url}")
    print("=" * 70 + "\n")

    try:
        worker = stream_mgr.attach_icsee_camera(
            camera_id=icsee_cam_id,
            ip=icsee_ip,
            port=icsee_port,
            username=icsee_user,
            password=icsee_pass,
            stream_type=icsee_stream,
            name=f"ICSee IP Camera ({icsee_ip})"
        )
        logger.info(f"Successfully attached ICSee camera worker on {icsee_cam_id}")
    except Exception as ex:
        logger.error(f"Error attaching ICSee camera feed: {ex}", exc_info=True)


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
