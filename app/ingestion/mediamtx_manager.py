"""MediaMTX Streaming Layer Controller for 600-Camera Scalable Deployment.

Provides zero-copy RTSP stream ingestion, dynamic REST path management,
and browser-ready WebRTC (WHEP) & LL-HLS URL generation.
"""

import os
import sys
import time
import json
import logging
import subprocess
import threading
from typing import Dict, Any, Optional, List
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


class MediaMTXManager:
    """Manages the lifecycle, dynamic paths, and status of the MediaMTX streaming server."""

    def __init__(
        self,
        api_url: str = "http://localhost:9997",
        config_path: Optional[str] = None,
        binary_path: Optional[str] = None
    ):
        self.api_url = api_url.rstrip("/")
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config_path = config_path or os.path.join(base_dir, "config", "mediamtx.yml")
        self.binary_path = binary_path or os.path.join(base_dir, "scripts", "mediamtx")
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._registered_paths: Dict[str, str] = {}
        self._is_running = False

    def is_api_alive(self) -> bool:
        """Check if MediaMTX REST API responds."""
        try:
            req = urllib.request.Request(f"{self.api_url}/v3/paths/list", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def start_daemon(self) -> bool:
        """Ensure MediaMTX is active. Spawns local supervisor if not running."""
        with self._lock:
            if self.is_api_alive():
                logger.info("MediaMTX server is already running and accessible via REST API.")
                self._is_running = True
                return True

            if not os.path.isfile(self.binary_path):
                logger.warning(f"MediaMTX binary not found at {self.binary_path}. Running without embedded supervisor.")
                return False

            if not os.path.isfile(self.config_path):
                logger.warning(f"MediaMTX config not found at {self.config_path}.")
                return False

            try:
                cmd = [self.binary_path, self.config_path]
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid if hasattr(os, "setsid") else None
                )
                logger.info(f"Spawned MediaMTX daemon process (PID: {self._process.pid})")
                # Wait briefly for startup
                for _ in range(20):
                    time.sleep(0.1)
                    if self.is_api_alive():
                        self._is_running = True
                        logger.info("MediaMTX daemon initialized successfully.")
                        return True
                return False
            except Exception as ex:
                logger.error(f"Failed to start MediaMTX daemon: {ex}")
                return False

    def stop_daemon(self):
        """Stop local daemon if spawned by this manager."""
        with self._lock:
            if self._process is not None:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=2.0)
                except Exception:
                    try:
                        self._process.kill()
                    except Exception:
                        pass
                self._process = None
                self._is_running = False

    def _normalize_name(self, camera_id: str) -> str:
        """Standardize camera identifier for MediaMTX path naming (e.g. CAM-001 -> cam-001)."""
        return camera_id.strip().lower().replace("_", "-")

    def add_or_update_path(
        self,
        camera_id: str,
        source_url: str,
        source_on_demand: bool = True
    ) -> bool:
        """Dynamically register or update an RTSP source in MediaMTX via REST API."""
        path_name = self._normalize_name(camera_id)
        payload = {
            "source": source_url,
            "sourceOnDemand": source_on_demand,
            "sourceOnDemandStartTimeout": "10s",
            "sourceOnDemandCloseAfter": "15s"
        }

        with self._lock:
            self._registered_paths[path_name] = source_url

        if not self.is_api_alive():
            # If daemon not yet up, attempt to start
            self.start_daemon()

        # Try adding path
        url_add = f"{self.api_url}/v3/config/paths/add/{path_name}"
        data = json.dumps(payload).encode("utf-8")
        try:
            req = urllib.request.Request(
                url_add,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status in (200, 201):
                    logger.debug(f"MediaMTX path [{path_name}] added for source: {source_url}")
                    return True
        except urllib.error.HTTPError as he:
            if he.code == 400:
                # Path might already exist; update it with patch
                try:
                    url_patch = f"{self.api_url}/v3/config/paths/patch/{path_name}"
                    req_patch = urllib.request.Request(
                        url_patch,
                        data=data,
                        headers={"Content-Type": "application/json"},
                        method="PATCH"
                    )
                    with urllib.request.urlopen(req_patch, timeout=2.0) as patch_resp:
                        return patch_resp.status in (200, 204)
                except Exception as patch_err:
                    logger.debug(f"MediaMTX patch error on [{path_name}]: {patch_err}")
            else:
                logger.debug(f"MediaMTX HTTPError adding path [{path_name}]: {he}")
        except Exception as ex:
            logger.debug(f"MediaMTX error adding path [{path_name}]: {ex}")

        return False

    def remove_path(self, camera_id: str) -> bool:
        """Remove a dynamic path from MediaMTX."""
        path_name = self._normalize_name(camera_id)
        with self._lock:
            self._registered_paths.pop(path_name, None)

        try:
            url_del = f"{self.api_url}/v3/config/paths/delete/{path_name}"
            req = urllib.request.Request(url_del, method="DELETE")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return resp.status in (200, 204)
        except Exception:
            return False

    def get_stream_urls(self, camera_id: str, client_host: str = "localhost") -> Dict[str, str]:
        """Generate browser WebRTC, HLS, and RTSP stream endpoints."""
        path_name = self._normalize_name(camera_id)
        clean_host = client_host.split(":")[0] if ":" in client_host else client_host
        if not clean_host or clean_host in ("0.0.0.0", "127.0.0.1"):
            clean_host = "localhost"

        return {
            "camera_id": camera_id,
            "path_name": path_name,
            "webrtc_url": f"http://{clean_host}:8889/{path_name}/whep",
            "hls_url": f"http://{clean_host}:8888/{path_name}/index.m3u8",
            "rtsp_url": f"rtsp://{clean_host}:8554/{path_name}",
            "is_mediamtx_active": self.is_api_alive()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Query MediaMTX paths and viewer statistics."""
        active_count = len(self._registered_paths)
        mediamtx_paths: List[Dict[str, Any]] = []
        is_live = self.is_api_alive()

        if is_live:
            try:
                req = urllib.request.Request(f"{self.api_url}/v3/paths/list", method="GET")
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        mediamtx_paths = data.get("items", [])
            except Exception:
                pass

        return {
            "is_mediamtx_live": is_live,
            "registered_paths_count": active_count,
            "active_streams_in_server": len(mediamtx_paths),
            "mediamtx_api_url": self.api_url
        }


# Global Singleton instance
mediamtx_mgr = MediaMTXManager()
