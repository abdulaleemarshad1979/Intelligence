"""Spatio-Temporal Topology Graph (STTG) G = (V, E).

Models city-wide surveillance nodes as topological vertices and pedestrian street corridors as edges.
Calculates multi-modal connection weights combining:
1. Re-ID metric visual cosine similarity S_reid
2. PAR discrete semantic attribute Jaccard similarity S_attr
3. Physical kinematic transit probability P_transition
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np

from app.graph.kinematics import KinematicValidator
from app.vision.par_engine import AttributeParsingEngine
from app.vision.reid_engine import OSNetReIDEngine

logger = logging.getLogger(__name__)


@dataclass
class STTGNode:
    tracklet_id: str
    camera_id: str
    t_in: float
    t_out: float
    embedding: List[float]
    active_attributes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class STTGEdge:
    source_tracklet_id: str
    target_tracklet_id: str
    from_camera_id: str
    to_camera_id: str
    delta_t: float
    reid_similarity: float
    attr_similarity: float
    transition_prob: float
    composite_weight: float


class SpatioTemporalTopologyGraph:
    """Municipal-scale Spatio-Temporal Topology Graph."""

    def __init__(self, kinematic_validator: Optional[KinematicValidator] = None):
        self.kinematics = kinematic_validator or KinematicValidator()
        self.nodes: Dict[str, STTGNode] = {}
        self.cameras: Dict[str, Dict[str, Any]] = {}
        self._init_default_municipal_topology()

    def _init_default_municipal_topology(self):
        """Initializes default municipal topology with known surveillance corridors."""
        default_distances = {
            ("CAM-001", "CAM-002"): 120.0,
            ("CAM-002", "CAM-003"): 150.0,
            ("CAM-001", "CAM-004"): 200.0,
            ("CAM-004", "CAM-005"): 180.0,
            ("CAM-003", "CAM-005"): 250.0,
            ("CAM-001", "CAM-017"): 90.0,
            ("CAM-017", "CAM-002"): 110.0,
        }
        for (c1, c2), dist in default_distances.items():
            self.kinematics.set_distance(c1, c2, dist, bidirectional=True)

    def add_camera(self, camera_id: str, name: str, lat: float, lon: float, zone: str = "Central Zone"):
        self.cameras[camera_id] = {
            "camera_id": camera_id,
            "name": name,
            "lat": lat,
            "lon": lon,
            "zone": zone
        }

    def register_tracklet(
        self,
        tracklet_id: str,
        camera_id: str,
        t_in: float,
        t_out: float,
        embedding: List[float],
        active_attributes: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> STTGNode:
        node = STTGNode(
            tracklet_id=tracklet_id,
            camera_id=camera_id,
            t_in=t_in,
            t_out=t_out,
            embedding=embedding,
            active_attributes=active_attributes or [],
            metadata=metadata or {}
        )
        self.nodes[tracklet_id] = node
        return node

    def compute_edge_weight(self, source_node: STTGNode, target_node: STTGNode) -> Optional[STTGEdge]:
        """Calculates directed connection weight W(T_i, T_j) = S_reid * S_attr * P_transition."""
        delta_t = target_node.t_in - source_node.t_out
        if delta_t < 0:
            return None  # Causality violation (cannot travel backward in standard forward edge)

        p_trans = self.kinematics.compute_transition_probability(
            source_node.camera_id, target_node.camera_id, delta_t
        )
        if p_trans <= 0.0:
            return None  # Kinematically pruned

        # Re-ID visual similarity
        sim_reid = OSNetReIDEngine.compute_cosine_similarity(source_node.embedding, target_node.embedding)

        # Attribute Jaccard similarity
        sim_attr = AttributeParsingEngine.compute_jaccard_similarity(
            source_node.active_attributes, target_node.active_attributes
        )

        composite_w = float(sim_reid * sim_attr * p_trans)

        return STTGEdge(
            source_tracklet_id=source_node.tracklet_id,
            target_tracklet_id=target_node.tracklet_id,
            from_camera_id=source_node.camera_id,
            to_camera_id=target_node.camera_id,
            delta_t=delta_t,
            reid_similarity=round(sim_reid, 4),
            attr_similarity=round(sim_attr, 4),
            transition_prob=round(p_trans, 4),
            composite_weight=round(composite_w, 4)
        )

    def export_topology_for_gis(self) -> Dict[str, Any]:
        """Exports graph topology for frontend Leaflet/Mapbox visualization."""
        nodes_list = []
        for cid, meta in self.cameras.items():
            nodes_list.append({
                "id": cid,
                "name": meta.get("name", cid),
                "lat": meta.get("lat", 16.9890),
                "lon": meta.get("lon", 82.2470),
                "zone": meta.get("zone", "Central Zone")
            })

        edges_list = []
        for from_cam, neighbors in self.kinematics.camera_distances.items():
            for to_cam, dist in neighbors.items():
                if from_cam < to_cam:  # Deduplicate undirected pairs
                    edges_list.append({
                        "from": from_cam,
                        "to": to_cam,
                        "distance_m": dist,
                        "t_min_sec": round(dist / self.kinematics.v_max, 1),
                        "t_max_sec": round(dist / self.kinematics.v_min, 1)
                    })

        return {
            "nodes": nodes_list,
            "edges": edges_list,
            "total_registered_tracklets": len(self.nodes)
        }
