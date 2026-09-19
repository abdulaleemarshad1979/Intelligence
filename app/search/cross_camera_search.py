"""Cross-Camera Topological Search Engine.

Traverses the camera graph topology to discover reachable downstream CCTV nodes
along heading trajectories.
"""

from typing import Dict, Any, List, Optional, Tuple
import math

def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

class CrossCameraSearchEngine:
    """Explores topological network connections between CCTV cameras."""

    def __init__(self, camera_registry: Dict[str, Any]):
        self.camera_registry = camera_registry

    def get_distance_meters(self, cam_a_id: str, cam_b_id: str) -> float:
        if cam_a_id == cam_b_id:
            return 0.0

        cam_a = self.camera_registry.get(cam_a_id)
        cam_b = self.camera_registry.get(cam_b_id)

        if cam_a and cam_b:
            lat1, lon1 = float(cam_a.get("latitude", 16.99)), float(cam_a.get("longitude", 82.24))
            lat2, lon2 = float(cam_b.get("latitude", 16.99)), float(cam_b.get("longitude", 82.24))
            dist = haversine_distance_meters(lat1, lon1, lat2, lon2)
            return max(50.0, dist)
        return 400.0  # Default nominal distance between urban blocks

    def get_downstream_cameras(self, origin_camera_id: str, max_hops: int = 3) -> List[str]:
        visited = set()
        queue = [(origin_camera_id, 0)]
        reachable = []

        while queue:
            curr_id, hops = queue.pop(0)
            if curr_id not in visited:
                visited.add(curr_id)
                if curr_id != origin_camera_id:
                    reachable.append(curr_id)

                if hops < max_hops:
                    info = self.camera_registry.get(curr_id, {})
                    adj = info.get("adjacent_cameras", [])
                    for nxt in adj:
                        if nxt not in visited:
                            queue.append((nxt, hops + 1))

        return reachable
