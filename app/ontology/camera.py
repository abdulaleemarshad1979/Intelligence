"""Investigation Ontology: Camera Node.

Represents a georeferenced CCTV node within the city-wide camera topology graph,
including geospatial coordinates, physical mount parameters, and topological adjacencies.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class CameraTopologyLink:
    target_camera_id: str
    distance_meters: float
    min_transit_seconds: float
    max_transit_seconds: float
    typical_direction: str = "NORTH"

@dataclass
class CameraObject:
    """Georeferenced CCTV Node in the investigation network."""
    camera_id: str
    name: str
    latitude: float
    longitude: float
    zone: str = "Central Division"
    view_direction: str = "NORTH"
    mounting_height_m: float = 4.5
    tilt_angle_deg: float = 35.0
    connected_topology: List[CameraTopologyLink] = field(default_factory=list)
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "zone": self.zone,
            "view_direction": self.view_direction,
            "mounting_height_m": self.mounting_height_m,
            "tilt_angle_deg": self.tilt_angle_deg,
            "connected_topology": [
                {
                    "target_camera_id": link.target_camera_id,
                    "distance_meters": link.distance_meters,
                    "min_transit_seconds": link.min_transit_seconds,
                    "max_transit_seconds": link.max_transit_seconds,
                    "typical_direction": link.typical_direction
                }
                for link in self.connected_topology
            ],
            "is_active": self.is_active
        }
