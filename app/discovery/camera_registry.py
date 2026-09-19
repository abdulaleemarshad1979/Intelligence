"""Camera Registry and Lifecycle Management.

Auto-registers discovered ONVIF camera streams into the database/repository,
handles deduplication by MAC address/IP, municipal surveillance zone assignment,
and approval/rejection workflows for law enforcement operations.
"""

import logging
from typing import List, Dict, Any, Optional
from app.database.repository import Repository
from app.database.models import CameraEntity
from app.discovery.onvif_scanner import DiscoveredCamera

logger = logging.getLogger(__name__)


class CameraRegistry:
    """Enterprise camera registry managing auto-registration, GIS mapping, and approval."""

    def __init__(self, repository: Optional[Repository] = None):
        self.repository = repository or Repository()

    def register_or_update(self, camera_data: Dict[str, Any]) -> CameraEntity:
        """Register a new camera or update an existing camera by camera_id or MAC/IP."""
        camera_id = camera_data.get("camera_id") or f"CAM-{camera_data.get('ip_address', '127.0.0.1').replace('.', '-')}"
        existing = self.repository.get_camera_by_id(camera_id)

        # Check by MAC address if available
        mac = camera_data.get("mac_address", "").strip()
        if not existing and mac:
            all_cams = self.repository.get_all_cameras()
            for c in all_cams:
                if c.mac_address and c.mac_address.upper() == mac.upper():
                    existing = c
                    camera_id = c.camera_id
                    break

        if existing:
            # Update dynamic parameters while respecting approval state
            existing.ip_address = camera_data.get("ip_address", existing.ip_address)
            existing.rtsp_url = camera_data.get("rtsp_url", existing.rtsp_url)
            existing.manufacturer = camera_data.get("manufacturer", existing.manufacturer)
            existing.model_name = camera_data.get("model_name", existing.model_name)
            if mac:
                existing.mac_address = mac
            self.repository.save_camera(existing)
            return existing

        # Create new unapproved discovered camera
        new_cam = CameraEntity(
            camera_id=camera_id,
            name=camera_data.get("name") or f"{camera_data.get('manufacturer', 'ONVIF')} {camera_data.get('model_name', 'Cam')} ({camera_data.get('ip_address')})",
            latitude=float(camera_data.get("latitude", 0.0)),
            longitude=float(camera_data.get("longitude", 0.0)),
            zone=camera_data.get("zone", "Unassigned Zone"),
            view_direction=camera_data.get("view_direction", "NORTH"),
            connected_topology=camera_data.get("connected_topology", []),
            is_active=bool(camera_data.get("is_active", False)),
            ip_address=camera_data.get("ip_address", "127.0.0.1"),
            rtsp_url=camera_data.get("rtsp_url", ""),
            manufacturer=camera_data.get("manufacturer", "Generic ONVIF"),
            model_name=camera_data.get("model_name", "IP Camera"),
            mac_address=mac,
            discovery_status="DISCOVERED"
        )
        self.repository.save_camera(new_cam)
        logger.info(f"Auto-registered new camera: {camera_id} from {new_cam.ip_address}")
        return new_cam

    def auto_register_discovered(self, cam: DiscoveredCamera) -> CameraEntity:
        """Helper to register a DiscoveredCamera instance."""
        data = {
            "camera_id": cam.camera_id,
            "ip_address": cam.ip_address,
            "rtsp_url": cam.rtsp_url,
            "manufacturer": cam.manufacturer,
            "model_name": cam.model_name,
            "mac_address": cam.mac_address,
            "discovery_status": cam.discovery_status,
            "is_active": False
        }
        return self.register_or_update(data)

    def batch_register(self, camera_list: List[Any]) -> List[CameraEntity]:
        """Batch process discovered camera list."""
        results = []
        for item in camera_list:
            if isinstance(item, DiscoveredCamera):
                results.append(self.auto_register_discovered(item))
            elif isinstance(item, dict):
                results.append(self.register_or_update(item))
        return results

    def approve_camera(
        self,
        camera_id: str,
        name: str,
        latitude: float,
        longitude: float,
        zone: str = "Central Zone",
        view_direction: str = "NORTH"
    ) -> bool:
        """Promote a discovered camera to active surveillance topology."""
        ok = self.repository.approve_discovered_camera(
            camera_id=camera_id,
            name=name,
            latitude=latitude,
            longitude=longitude,
            zone=zone
        )
        if ok:
            cam = self.repository.get_camera_by_id(camera_id)
            if cam:
                cam.view_direction = view_direction
                self.repository.save_camera(cam)
        return ok

    def reject_camera(self, camera_id: str, reason: str = "") -> bool:
        """Reject and decommission an unauthorized camera from the network."""
        cam = self.repository.get_camera_by_id(camera_id)
        if not cam:
            return False
        cam.discovery_status = "REJECTED"
        cam.is_active = False
        self.repository.save_camera(cam)
        logger.warning(f"Camera {camera_id} rejected: {reason}")
        return True

    def get_camera(self, camera_id: str) -> Optional[CameraEntity]:
        return self.repository.get_camera_by_id(camera_id)

    def list_cameras(
        self,
        status: Optional[str] = None,
        zone: Optional[str] = None,
        active_only: bool = False
    ) -> List[CameraEntity]:
        cameras = self.repository.get_all_cameras()
        if status:
            cameras = [c for c in cameras if c.discovery_status == status]
        if zone:
            cameras = [c for c in cameras if c.zone.lower() == zone.lower()]
        if active_only:
            cameras = [c for c in cameras if c.is_active]
        return cameras


camera_registry = CameraRegistry()
