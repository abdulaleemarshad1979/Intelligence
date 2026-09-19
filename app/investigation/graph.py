"""Investigation Graph Engine.

Builds graph topologies (Nodes: Incidents, Persons, Tracks, Cameras, Objects;
Edges: Associated links, transit steps, observations) for interactive visualization.
"""

from typing import Dict, Any, List, Optional
import time

class InvestigationGraphBuilder:
    """Builds node-and-edge graphs for investigative visualization."""

    def build_graph(
        self,
        incident: Dict[str, Any],
        candidate_associations: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        nodes = []
        edges = []
        seen_nodes = set()

        def add_node(node_id: str, label: str, node_type: str, metadata: Dict[str, Any]):
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                nodes.append({
                    "id": node_id,
                    "label": label,
                    "type": node_type,
                    "metadata": metadata
                })

        def add_edge(source: str, target: str, label: str, edge_type: str, weight: float = 1.0, metadata: Dict[str, Any] = None):
            edges.append({
                "id": f"e_{source}_{target}",
                "source": source,
                "target": target,
                "label": label,
                "type": edge_type,
                "weight": weight,
                "metadata": metadata or {}
            })

        # 1. Incident Node
        inc_id = incident.get("incident_id", "INC-001")
        case_no = incident.get("case_number", "INC-2026-0041")
        add_node(inc_id, case_no, "INCIDENT", incident)

        # 2. Origin Camera Node
        origin_cam = incident.get("camera_id", "CAM-017")
        add_node(origin_cam, origin_cam, "CAMERA", {"name": f"Origin CCTV {origin_cam}"})
        add_edge(inc_id, origin_cam, "OCCURRED_AT", "SPATIAL_LOCATION")

        # 3. Seed Track Node
        seed_track_id = incident.get("seed_track_id")
        if seed_track_id:
            add_node(seed_track_id, f"Track {seed_track_id}", "TRACK", {
                "camera_id": origin_cam,
                "role": "SEED_TARGET"
            })
            add_edge(inc_id, seed_track_id, "SEED_PROBE", "INVESTIGATION_TARGET")
            add_edge(seed_track_id, origin_cam, "CAPTURED_ON", "OBSERVATION_FEED")

        # 4. Candidate Associations
        prev_track = seed_track_id
        for assoc in candidate_associations:
            cand_id = assoc.get("candidate_track_id")
            cand_cam = assoc.get("candidate_camera")
            if not cand_id:
                continue

            add_node(cand_cam, cand_cam, "CAMERA", {"name": f"CCTV Node {cand_cam}"})
            add_node(cand_id, f"Track {cand_id}", "TRACK", {
                "camera_id": cand_cam,
                "time": assoc.get("candidate_time"),
                "status": assoc.get("association_status", "CANDIDATE"),
                "narrative": assoc.get("qualitative_narrative")
            })
            add_edge(cand_id, cand_cam, "CAPTURED_ON", "OBSERVATION_FEED")

            # Link with previous track or probe track
            source_track = assoc.get("probe_track_id") or prev_track
            if source_track:
                edge_label = f"Match ({assoc.get('composite_score', 0.0)*100:.0f}%)"
                status = assoc.get("association_status", "CANDIDATE")
                edge_type = "CONFIRMED_LINK" if status == "CONFIRMED" else "CANDIDATE_LINK"
                add_edge(
                    source_track,
                    cand_id,
                    edge_label,
                    edge_type,
                    weight=assoc.get("composite_score", 0.5),
                    metadata={
                        "narrative": assoc.get("qualitative_narrative"),
                        "evidence": assoc.get("evidence_breakdown")
                    }
                )

            # Camera-to-Camera transit topology edge
            source_cam = assoc.get("probe_camera")
            if source_cam and cand_cam and source_cam != cand_cam:
                add_edge(source_cam, cand_cam, "TRANSIT_ROUTE", "CAMERA_TOPOLOGY", weight=0.5)

            prev_track = cand_id

        # 5. Add custom confirmed relationships
        for rel in relationships:
            s_id = rel.get("source_id")
            t_id = rel.get("target_id")
            if s_id in seen_nodes and t_id in seen_nodes:
                add_edge(s_id, t_id, rel.get("relationship_type", "LINK"), "EXPLICIT_RELATIONSHIP", metadata=rel)

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "tracks_count": sum(1 for n in nodes if n["type"] == "TRACK"),
                "cameras_count": sum(1 for n in nodes if n["type"] == "CAMERA")
            }
        }
