"""Discovery REST API Routes for Zero-Touch ONVIF Ingestion."""

import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.discovery.onvif_scanner import onvif_scanner, ONVIFDiscoveryEngine
from app.discovery.camera_registry import camera_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/discovery", tags=["Camera Discovery"])


class ScanNetworkPayload(BaseModel):
    timeout_seconds: Optional[float] = 1.5
    include_simulated_if_empty: Optional[bool] = True


class ApproveCameraPayload(BaseModel):
    camera_id: str
    name: str
    latitude: float
    longitude: float
    zone: Optional[str] = "East Zone"
    view_direction: Optional[str] = "NORTH"


class RejectCameraPayload(BaseModel):
    camera_id: str
    reason: Optional[str] = "Unauthorized network device"


@router.post("/scan")
def scan_network_endpoint(payload: ScanNetworkPayload):
    """Broadcasts ONVIF WS-Discovery probe across the surveillance VLAN."""
    discovered = onvif_scanner.scan_network(
        timeout=payload.timeout_seconds,
        include_simulated_if_empty=payload.include_simulated_if_empty
    )
    # Auto-register newly discovered cameras into registry
    registered = camera_registry.batch_register(discovered)

    return {
        "status": "SUCCESS",
        "found_count": len(discovered),
        "registered_count": len(registered),
        "cameras": [c.to_dict() if hasattr(c, "to_dict") else c for c in discovered]
    }


@router.get("/cameras")
def list_discovered_cameras(status: Optional[str] = None, zone: Optional[str] = None):
    """List all registered and discovered cameras."""
    cams = camera_registry.list_cameras(status=status, zone=zone)
    return {
        "status": "SUCCESS",
        "total": len(cams),
        "cameras": [
            {
                "camera_id": c.camera_id,
                "name": c.name,
                "ip_address": c.ip_address,
                "rtsp_url": c.rtsp_url,
                "manufacturer": c.manufacturer,
                "model_name": c.model_name,
                "mac_address": c.mac_address,
                "latitude": c.latitude,
                "longitude": c.longitude,
                "zone": c.zone,
                "status": c.discovery_status,
                "is_active": c.is_active
            }
            for c in cams
        ]
    }


@router.post("/approve")
def approve_camera_endpoint(payload: ApproveCameraPayload):
    """Approve a discovered camera, assigning geographic coordinates and zone."""
    ok = camera_registry.approve_camera(
        camera_id=payload.camera_id,
        name=payload.name,
        latitude=payload.latitude,
        longitude=payload.longitude,
        zone=payload.zone or "Central Zone",
        view_direction=payload.view_direction or "NORTH"
    )
    if not ok:
        raise HTTPException(status_code=404, detail=f"Camera {payload.camera_id} not found")

    cam = camera_registry.get_camera(payload.camera_id)
    return {
        "status": "SUCCESS",
        "message": f"Camera {payload.camera_id} approved and added to active surveillance topology.",
        "camera": {
            "camera_id": cam.camera_id,
            "name": cam.name,
            "latitude": cam.latitude,
            "longitude": cam.longitude,
            "zone": cam.zone,
            "status": cam.discovery_status,
            "is_active": cam.is_active
        }
    }


@router.post("/reject")
def reject_camera_endpoint(payload: RejectCameraPayload):
    """Reject and decommission an unauthorized camera from the network."""
    ok = camera_registry.reject_camera(payload.camera_id, payload.reason or "")
    if not ok:
        raise HTTPException(status_code=404, detail=f"Camera {payload.camera_id} not found")
    return {
        "status": "SUCCESS",
        "message": f"Camera {payload.camera_id} marked as REJECTED."
    }


@router.get("/status")
def get_discovery_status():
    """Returns discovery service status and probe parameters."""
    return {
        "service": "ONVIF_WS_DISCOVERY",
        "multicast_group": ONVIFDiscoveryEngine.MULTICAST_GROUP,
        "multicast_port": ONVIFDiscoveryEngine.MULTICAST_PORT,
        "status": "LISTENING"
    }
