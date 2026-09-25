"""Database repository operations for criminal records, tracks, and matches."""

import json
import sqlite3
from typing import List, Optional, Dict, Any
from app.database.database import get_db_connection
from app.database.models import (
    CriminalRecord, TrackObservation, MatchEvent,
    PersonTarget, CameraEntity, IncidentCase,
    ObservationRecord, FeatureRecord, RelationshipLink, AdjudicationReview
)

class Repository:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def insert_criminal_record(self, record: CriminalRecord) -> bool:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            carried_json = json.dumps(getattr(record, "carried_objects", []))
            enhanced_url = getattr(record, "enhanced_photo_url", "")
            cursor.execute("""
            INSERT OR REPLACE INTO criminal_records (
                id, fir_no, unit_name, subdivision, police_station, accused_name,
                alias, age, gender, acts_sec, brief_facts, latitude, longitude,
                status_of_case, photo_url, known_height_cm, torso_leg_ratio,
                stride_length_cm, posture_lean_angle, posture_correctness,
                clothing_upper_color, clothing_lower_color,
                face_embedding, body_embedding, gait_embedding,
                carried_objects, enhanced_photo_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.id, record.fir_no, record.unit_name, record.subdivision, record.police_station,
                record.accused_name, record.alias, record.age, record.gender, record.acts_sec,
                record.brief_facts, record.latitude, record.longitude, record.status_of_case,
                record.photo_url, record.known_height_cm, record.torso_leg_ratio, record.stride_length_cm,
                record.posture_lean_angle, record.posture_correctness, record.clothing_upper_color,
                record.clothing_lower_color, json.dumps(record.face_embedding),
                json.dumps(record.body_embedding), json.dumps(record.gait_embedding),
                carried_json, enhanced_url
            ))
            conn.commit()
            return True
        finally:
            conn.close()

    def get_all_criminal_records(self) -> List[CriminalRecord]:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM criminal_records ORDER BY created_at DESC")
            rows = cursor.fetchall()
            records = []
            for r in rows:
                rec = CriminalRecord(
                    id=r["id"],
                    fir_no=r["fir_no"],
                    unit_name=r["unit_name"],
                    subdivision=r["subdivision"],
                    police_station=r["police_station"],
                    accused_name=r["accused_name"],
                    alias=r["alias"] or "",
                    age=r["age"] or 0,
                    gender=r["gender"] or "Male",
                    acts_sec=r["acts_sec"] or "",
                    brief_facts=r["brief_facts"] or "",
                    latitude=r["latitude"] or 0.0,
                    longitude=r["longitude"] or 0.0,
                    status_of_case=r["status_of_case"] or "",
                    photo_url=r["photo_url"] or "",
                    known_height_cm=r["known_height_cm"] or 170.0,
                    torso_leg_ratio=r["torso_leg_ratio"] or 0.85,
                    stride_length_cm=r["stride_length_cm"] or 65.0,
                    posture_lean_angle=r["posture_lean_angle"] or 0.0,
                    posture_correctness=r["posture_correctness"] or 0.85,
                    clothing_upper_color=r["clothing_upper_color"] or "#334455",
                    clothing_lower_color=r["clothing_lower_color"] or "#112233",
                    carried_objects=json.loads(r["carried_objects"]) if ("carried_objects" in r.keys() and r["carried_objects"]) else [],
                    enhanced_photo_url=r["enhanced_photo_url"] if ("enhanced_photo_url" in r.keys() and r["enhanced_photo_url"]) else "",
                    face_embedding=json.loads(r["face_embedding"]) if r["face_embedding"] else [],
                    body_embedding=json.loads(r["body_embedding"]) if r["body_embedding"] else [],
                    gait_embedding=json.loads(r["gait_embedding"]) if r["gait_embedding"] else []
                )
                records.append(rec)
            return records
        finally:
            conn.close()

    def get_criminal_record_by_id(self, record_id: str) -> Optional[CriminalRecord]:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM criminal_records WHERE id = ?", (record_id,))
            r = cursor.fetchone()
            if not r:
                return None
            return CriminalRecord(
                id=r["id"],
                fir_no=r["fir_no"],
                unit_name=r["unit_name"],
                subdivision=r["subdivision"],
                police_station=r["police_station"],
                accused_name=r["accused_name"],
                alias=r["alias"] or "",
                age=r["age"] or 0,
                gender=r["gender"] or "Male",
                acts_sec=r["acts_sec"] or "",
                brief_facts=r["brief_facts"] or "",
                latitude=r["latitude"] or 0.0,
                longitude=r["longitude"] or 0.0,
                status_of_case=r["status_of_case"] or "",
                photo_url=r["photo_url"] or "",
                known_height_cm=r["known_height_cm"] or 170.0,
                torso_leg_ratio=r["torso_leg_ratio"] or 0.85,
                stride_length_cm=r["stride_length_cm"] or 65.0,
                posture_lean_angle=r["posture_lean_angle"] or 0.0,
                posture_correctness=r["posture_correctness"] or 0.85,
                clothing_upper_color=r["clothing_upper_color"] or "#334455",
                clothing_lower_color=r["clothing_lower_color"] or "#112233",
                carried_objects=json.loads(r["carried_objects"]) if ("carried_objects" in r.keys() and r["carried_objects"]) else [],
                enhanced_photo_url=r["enhanced_photo_url"] if ("enhanced_photo_url" in r.keys() and r["enhanced_photo_url"]) else "",
                face_embedding=json.loads(r["face_embedding"]) if r["face_embedding"] else [],
                body_embedding=json.loads(r["body_embedding"]) if r["body_embedding"] else [],
                gait_embedding=json.loads(r["gait_embedding"]) if r["gait_embedding"] else []
            )
        finally:
            conn.close()

    def save_track(self, track: TrackObservation) -> bool:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT OR REPLACE INTO tracks (
                track_id, camera_id, first_seen, last_seen, frame_count,
                best_frame_path, face_visible, face_status, face_tier_details,
                estimated_height_cm, body_proportions, clothing_upper, clothing_lower,
                stride_length_px, stride_length_cm, cadence_steps_per_sec,
                spine_tilt_deg, posture_score, gait_wave,
                face_embedding, body_embedding, gait_embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                track.track_id, track.camera_id, track.first_seen, track.last_seen, track.frame_count,
                track.best_frame_path, 1 if track.face_visible else 0, track.face_status,
                json.dumps(track.face_tier_details), track.estimated_height_cm,
                json.dumps(track.body_proportions), track.clothing_upper, track.clothing_lower,
                track.stride_length_px, track.stride_length_cm, track.cadence_steps_per_sec,
                track.spine_tilt_deg, track.posture_score, json.dumps(track.gait_wave),
                json.dumps(track.face_embedding), json.dumps(track.body_embedding), json.dumps(track.gait_embedding)
            ))
            conn.commit()
            return True
        finally:
            conn.close()

    def get_all_tracks(self) -> List[Dict[str, Any]]:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM tracks ORDER BY last_seen DESC")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_track_by_id(self, track_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM tracks WHERE track_id = ?", (track_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d["face_tier_details"] = json.loads(d["face_tier_details"]) if d["face_tier_details"] else {}
                d["body_proportions"] = json.loads(d["body_proportions"]) if d["body_proportions"] else {}
                d["gait_wave"] = json.loads(d["gait_wave"]) if d["gait_wave"] else []
                return d
            return None
        finally:
            conn.close()

    def save_match_event(self, event: MatchEvent) -> bool:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT OR REPLACE INTO match_events (
                event_id, track_id, camera_id, suspect_id, suspect_name, fir_no,
                total_confidence, face_score, body_score, gait_score, height_score,
                is_face_available, status, evidence_breakdown, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id, event.track_id, event.camera_id, event.suspect_id, event.suspect_name,
                event.fir_no, event.total_confidence, event.face_score, event.body_score, event.gait_score,
                event.height_score, 1 if event.is_face_available else 0, event.status,
                json.dumps(event.evidence_breakdown), event.created_at
            ))
            conn.commit()
            return True
        finally:
            conn.close()

    def get_match_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM match_events ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            events = []
            for r in rows:
                d = dict(r)
                d["evidence_breakdown"] = json.loads(d["evidence_breakdown"]) if d["evidence_breakdown"] else {}
                events.append(d)
            return events
        finally:
            conn.close()

    def log_audit(self, action: str, details: str, operator: str = "OFFICER_IN_CHARGE"):
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            import time
            cursor.execute("INSERT INTO audit_logs (timestamp, action, details, operator) VALUES (?, ?, ?, ?)",
                           (time.time(), action, details, operator))
            conn.commit()
        finally:
            conn.close()

    # ==================== GOTHAM ONTOLOGY OPERATIONS ====================

    def save_person(self, person: PersonTarget) -> bool:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT OR REPLACE INTO persons (
                person_id, target_code, canonical_name, status, notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (person.person_id, person.target_code, person.canonical_name,
                  person.status, person.notes, person.created_at, person.updated_at))
            conn.commit()
            return True
        finally:
            conn.close()

    def get_person_by_id(self, person_id: str) -> Optional[PersonTarget]:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM persons WHERE person_id = ?", (person_id,))
            r = cursor.fetchone()
            if not r:
                return None
            return PersonTarget(
                person_id=r["person_id"],
                target_code=r["target_code"],
                canonical_name=r["canonical_name"],
                status=r["status"],
                notes=r["notes"] or "",
                created_at=r["created_at"] or 0.0,
                updated_at=r["updated_at"] or 0.0
            )
        finally:
            conn.close()

    def get_all_persons(self) -> List[PersonTarget]:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM persons ORDER BY updated_at DESC")
            rows = cursor.fetchall()
            return [
                PersonTarget(
                    person_id=r["person_id"],
                    target_code=r["target_code"],
                    canonical_name=r["canonical_name"],
                    status=r["status"],
                    notes=r["notes"] or "",
                    created_at=r["created_at"] or 0.0,
                    updated_at=r["updated_at"] or 0.0
                )
                for r in rows
            ]
        finally:
            conn.close()

    def save_incident(self, incident: IncidentCase) -> bool:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT OR REPLACE INTO incidents (
                incident_id, case_number, title, description, camera_id,
                incident_time, status, priority, officer_in_charge, seed_track_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (incident.incident_id, incident.case_number, incident.title,
                  incident.description, incident.camera_id, incident.incident_time,
                  incident.status, incident.priority, incident.officer_in_charge,
                  incident.seed_track_id, incident.created_at))
            conn.commit()
            return True
        finally:
            conn.close()

    def get_incident_by_id(self, incident_id: str) -> Optional[IncidentCase]:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM incidents WHERE incident_id = ?", (incident_id,))
            r = cursor.fetchone()
            if not r:
                return None
            return IncidentCase(
                incident_id=r["incident_id"],
                case_number=r["case_number"],
                title=r["title"],
                description=r["description"] or "",
                camera_id=r["camera_id"],
                incident_time=r["incident_time"],
                status=r["status"],
                priority=r["priority"],
                officer_in_charge=r["officer_in_charge"],
                seed_track_id=r["seed_track_id"],
                created_at=r["created_at"]
            )
        finally:
            conn.close()

    def get_all_incidents(self) -> List[IncidentCase]:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM incidents ORDER BY incident_time DESC")
            rows = cursor.fetchall()
            return [
                IncidentCase(
                    incident_id=r["incident_id"],
                    case_number=r["case_number"],
                    title=r["title"],
                    description=r["description"] or "",
                    camera_id=r["camera_id"],
                    incident_time=r["incident_time"],
                    status=r["status"],
                    priority=r["priority"],
                    officer_in_charge=r["officer_in_charge"],
                    seed_track_id=r["seed_track_id"],
                    created_at=r["created_at"]
                )
                for r in rows
            ]
        finally:
            conn.close()

    def save_observation(self, obs: ObservationRecord) -> bool:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT OR REPLACE INTO observations (
                observation_id, track_id, camera_id, timestamp,
                frame_number, crop_path, bbox_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (obs.observation_id, obs.track_id, obs.camera_id, obs.timestamp,
                  obs.frame_number, obs.crop_path, json.dumps(obs.bbox), obs.created_at))
            conn.commit()
            return True
        finally:
            conn.close()

    def get_observations_for_track(self, track_id: str) -> List[ObservationRecord]:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM observations WHERE track_id = ? ORDER BY timestamp ASC", (track_id,))
            rows = cursor.fetchall()
            return [
                ObservationRecord(
                    observation_id=r["observation_id"],
                    track_id=r["track_id"],
                    camera_id=r["camera_id"],
                    timestamp=r["timestamp"],
                    frame_number=r["frame_number"],
                    crop_path=r["crop_path"],
                    bbox=json.loads(r["bbox_json"]) if r["bbox_json"] else {},
                    created_at=r["created_at"]
                )
                for r in rows
            ]
        finally:
            conn.close()

    def save_feature(self, feat: FeatureRecord) -> bool:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT OR REPLACE INTO features (
                feature_id, track_id, observation_id, face_status,
                face_embedding, body_embedding, gait_embedding,
                pose_json, clothing_json, height_cm, carried_objects_json,
                direction, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (feat.feature_id, feat.track_id, feat.observation_id, feat.face_status,
                  json.dumps(feat.face_embedding), json.dumps(feat.body_embedding),
                  json.dumps(feat.gait_embedding), json.dumps(feat.pose_landmarks),
                  json.dumps(feat.clothing_attributes), feat.height_cm,
                  json.dumps(feat.carried_objects), feat.direction, feat.created_at))
            conn.commit()
            return True
        finally:
            conn.close()

    def get_feature_for_track(self, track_id: str) -> Optional[FeatureRecord]:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM features WHERE track_id = ? ORDER BY created_at DESC LIMIT 1", (track_id,))
            r = cursor.fetchone()
            if not r:
                return None
            return FeatureRecord(
                feature_id=r["feature_id"],
                track_id=r["track_id"],
                observation_id=r["observation_id"],
                face_status=r["face_status"] or "UNAVAILABLE",
                face_embedding=json.loads(r["face_embedding"]) if r["face_embedding"] else [],
                body_embedding=json.loads(r["body_embedding"]) if r["body_embedding"] else [],
                gait_embedding=json.loads(r["gait_embedding"]) if r["gait_embedding"] else [],
                pose_landmarks=json.loads(r["pose_json"]) if r["pose_json"] else {},
                clothing_attributes=json.loads(r["clothing_json"]) if r["clothing_json"] else {},
                height_cm=r["height_cm"] or 0.0,
                carried_objects=json.loads(r["carried_objects_json"]) if r["carried_objects_json"] else [],
                direction=r["direction"] or "NORTH",
                created_at=r["created_at"] or 0.0
            )
        finally:
            conn.close()

    def save_camera(self, cam: CameraEntity) -> bool:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT OR REPLACE INTO cameras (
                camera_id, name, latitude, longitude, zone, view_direction,
                connected_topology_json, is_active, ip_address, rtsp_url,
                manufacturer, model_name, mac_address, discovery_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cam.camera_id, cam.name, cam.latitude, cam.longitude, cam.zone,
                cam.view_direction, json.dumps(cam.connected_topology), 1 if cam.is_active else 0,
                cam.ip_address, cam.rtsp_url, cam.manufacturer, cam.model_name,
                cam.mac_address, cam.discovery_status
            ))
            conn.commit()
            return True
        finally:
            conn.close()

    def get_all_cameras(self) -> List[CameraEntity]:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM cameras WHERE is_active = 1")
            rows = cursor.fetchall()
            return [self._row_to_camera(r) for r in rows]
        finally:
            conn.close()

    def get_camera_by_id(self, camera_id: str) -> Optional[CameraEntity]:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM cameras WHERE camera_id = ?", (camera_id,))
            r = cursor.fetchone()
            if not r:
                return None
            return self._row_to_camera(r)
        finally:
            conn.close()

    def get_discovered_cameras(self, status: Optional[str] = None) -> List[CameraEntity]:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            if status:
                cursor.execute("SELECT * FROM cameras WHERE discovery_status = ? ORDER BY camera_id", (status,))
            else:
                cursor.execute("SELECT * FROM cameras ORDER BY camera_id")
            rows = cursor.fetchall()
            return [self._row_to_camera(r) for r in rows]
        finally:
            conn.close()

    def approve_discovered_camera(
        self,
        camera_id: str,
        name: str,
        latitude: float,
        longitude: float,
        zone: str = "East Zone"
    ) -> bool:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
            UPDATE cameras
            SET name = ?, latitude = ?, longitude = ?, zone = ?, discovery_status = 'APPROVED', is_active = 1
            WHERE camera_id = ?
            """, (name, latitude, longitude, zone, camera_id))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def _row_to_camera(self, r: Any) -> CameraEntity:
        # Gracefully handle row indexing across sqlite3.Row
        keys = r.keys() if hasattr(r, "keys") else []
        return CameraEntity(
            camera_id=r["camera_id"],
            name=r["name"],
            latitude=r["latitude"],
            longitude=r["longitude"],
            zone=r["zone"] if "zone" in keys and r["zone"] else "Central Division",
            view_direction=r["view_direction"] if "view_direction" in keys and r["view_direction"] else "NORTH",
            connected_topology=json.loads(r["connected_topology_json"]) if "connected_topology_json" in keys and r["connected_topology_json"] else [],
            is_active=bool(r["is_active"]),
            ip_address=r["ip_address"] if "ip_address" in keys and r["ip_address"] else "127.0.0.1",
            rtsp_url=r["rtsp_url"] if "rtsp_url" in keys and r["rtsp_url"] else "",
            manufacturer=r["manufacturer"] if "manufacturer" in keys and r["manufacturer"] else "Generic ONVIF",
            model_name=r["model_name"] if "model_name" in keys and r["model_name"] else "IP Camera",
            mac_address=r["mac_address"] if "mac_address" in keys and r["mac_address"] else "",
            discovery_status=r["discovery_status"] if "discovery_status" in keys and r["discovery_status"] else "APPROVED"
        )

    def save_relationship(self, rel: RelationshipLink) -> bool:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT OR REPLACE INTO relationships (
                relationship_id, source_type, source_id, target_type, target_id,
                relationship_type, confidence_score, evidence_json, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (rel.relationship_id, rel.source_type, rel.source_id,
                  rel.target_type, rel.target_id, rel.relationship_type,
                  rel.confidence_score, json.dumps(rel.evidence), rel.status, rel.created_at))
            conn.commit()
            return True
        finally:
            conn.close()

    def get_relationships(self, source_id: Optional[str] = None, target_id: Optional[str] = None) -> List[RelationshipLink]:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            if source_id and target_id:
                cursor.execute("SELECT * FROM relationships WHERE (source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)",
                               (source_id, target_id, target_id, source_id))
            elif source_id:
                cursor.execute("SELECT * FROM relationships WHERE source_id = ? OR target_id = ?", (source_id, source_id))
            else:
                cursor.execute("SELECT * FROM relationships ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [
                RelationshipLink(
                    relationship_id=r["relationship_id"],
                    source_type=r["source_type"],
                    source_id=r["source_id"],
                    target_type=r["target_type"],
                    target_id=r["target_id"],
                    relationship_type=r["relationship_type"],
                    confidence_score=r["confidence_score"] or 0.0,
                    evidence=json.loads(r["evidence_json"]) if r["evidence_json"] else {},
                    status=r["status"] or "CANDIDATE",
                    created_at=r["created_at"] or 0.0
                )
                for r in rows
            ]
        finally:
            conn.close()

    def update_relationship_status(self, relationship_id: str, new_status: str) -> bool:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("UPDATE relationships SET status = ? WHERE relationship_id = ?", (new_status, relationship_id))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def save_review(self, rev: AdjudicationReview) -> bool:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT OR REPLACE INTO reviews (
                review_id, relationship_id, target_id, reviewer_badge,
                reviewer_name, decision, review_notes, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (rev.review_id, rev.relationship_id, rev.target_id,
                  rev.reviewer_badge, rev.reviewer_name, rev.decision,
                  rev.review_notes, rev.timestamp))
            conn.commit()
            return True
        finally:
            conn.close()

    def get_reviews(self, target_id: Optional[str] = None) -> List[AdjudicationReview]:
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            if target_id:
                cursor.execute("SELECT * FROM reviews WHERE target_id = ? ORDER BY timestamp DESC", (target_id,))
            else:
                cursor.execute("SELECT * FROM reviews ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            return [
                AdjudicationReview(
                    review_id=r["review_id"],
                    relationship_id=r["relationship_id"],
                    target_id=r["target_id"],
                    reviewer_badge=r["reviewer_badge"],
                    reviewer_name=r["reviewer_name"],
                    decision=r["decision"],
                    review_notes=r["review_notes"] or "",
                    timestamp=r["timestamp"] or 0.0
                )
                for r in rows
            ]
        finally:
            conn.close()

    # ==================== SKELETON, GAIT DYNAMICS & MODEL TRAINING REPOSITORY ====================

    def save_person_skeleton(
        self,
        track_id: str,
        frame_idx: int,
        timestamp: float,
        bbox: List[int],
        keypoints_coco_17: List[Dict[str, Any]],
        keypoints_crop: Dict[str, Any],
        keypoints_global: Dict[str, Any],
        confidences: Dict[str, float],
        visible_joints_count: int,
        inter_ankle_dist: float,
        spine_tilt_deg: float,
        neck_point: List[float],
        is_confident: bool = True,
        source: str = "RTMPOSE_ONNX",
        person_id: Optional[str] = None
    ) -> bool:
        """Store fine-grained exo-skeleton keypoints for a single video frame."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT INTO person_skeletons (
                track_id, person_id, frame_idx, timestamp, bbox_json,
                keypoints_coco_json, keypoints_crop_json, keypoints_global_json,
                confidences_json, visible_joints_count, inter_ankle_dist,
                spine_tilt_deg, neck_point_json, is_confident, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                track_id, person_id or track_id, frame_idx, timestamp,
                json.dumps(bbox), json.dumps(keypoints_coco_17),
                json.dumps(keypoints_crop), json.dumps(keypoints_global),
                json.dumps(confidences), visible_joints_count, inter_ankle_dist,
                spine_tilt_deg, json.dumps(neck_point), 1 if is_confident else 0, source
            ))
            conn.commit()
            return True
        finally:
            conn.close()

    def save_person_skeletons_batch(self, skeletons: List[Dict[str, Any]]) -> int:
        """Batch insert frame-by-frame exo-skeletons."""
        if not skeletons:
            return 0
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            rows = [
                (
                    s["track_id"], s.get("person_id") or s["track_id"],
                    s["frame_idx"], s["timestamp"], json.dumps(s.get("bbox", [])),
                    json.dumps(s.get("keypoints_coco_17", [])),
                    json.dumps(s.get("keypoints_crop", {})),
                    json.dumps(s.get("keypoints_global", {})),
                    json.dumps(s.get("confidences", {})),
                    s.get("visible_joints_count", 0),
                    s.get("inter_ankle_dist", 0.0),
                    s.get("spine_tilt_deg", 0.0),
                    json.dumps(s.get("neck_point", [])),
                    1 if s.get("is_confident", True) else 0,
                    s.get("source", "RTMPOSE_ONNX")
                )
                for s in skeletons
            ]
            cursor.executemany("""
            INSERT INTO person_skeletons (
                track_id, person_id, frame_idx, timestamp, bbox_json,
                keypoints_coco_json, keypoints_crop_json, keypoints_global_json,
                confidences_json, visible_joints_count, inter_ankle_dist,
                spine_tilt_deg, neck_point_json, is_confident, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
            return len(rows)
        finally:
            conn.close()

    def get_skeletons_for_track(self, track_id: str) -> List[Dict[str, Any]]:
        """Retrieve temporal sequence of exo-skeleton frames for a person track."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
            SELECT * FROM person_skeletons WHERE track_id = ? ORDER BY frame_idx ASC
            """, (track_id,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                results.append({
                    "id": r["id"],
                    "track_id": r["track_id"],
                    "person_id": r["person_id"],
                    "frame_idx": r["frame_idx"],
                    "timestamp": r["timestamp"],
                    "bbox": json.loads(r["bbox_json"]) if r["bbox_json"] else [],
                    "keypoints_coco_17": json.loads(r["keypoints_coco_json"]) if r["keypoints_coco_json"] else [],
                    "keypoints_crop": json.loads(r["keypoints_crop_json"]) if r["keypoints_crop_json"] else {},
                    "keypoints_global": json.loads(r["keypoints_global_json"]) if r["keypoints_global_json"] else {},
                    "confidences": json.loads(r["confidences_json"]) if r["confidences_json"] else {},
                    "visible_joints_count": r["visible_joints_count"],
                    "inter_ankle_dist": r["inter_ankle_dist"],
                    "spine_tilt_deg": r["spine_tilt_deg"],
                    "neck_point": json.loads(r["neck_point_json"]) if r["neck_point_json"] else [],
                    "is_confident": bool(r["is_confident"]),
                    "source": r["source"]
                })
            return results
        finally:
            conn.close()

    def save_gait_dynamics(self, data: Dict[str, Any]) -> bool:
        """Store verified gait dynamics and kinematic waveforms for a track."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT OR REPLACE INTO gait_dynamics (
                track_id, person_id, sequence_length, stride_length_px,
                stride_length_cm, cadence_hz, spine_tilt_deg, posture_score,
                joint_velocities_json, fft_harmonics_json, gait_wave_json,
                gait_embedding_json, is_valid_gait, gait_usable
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["track_id"],
                data.get("person_id") or data["track_id"],
                data.get("sequence_length", 0),
                data.get("stride_length_px", 0.0),
                data.get("stride_length_cm", 0.0),
                data.get("cadence_hz", 0.0),
                data.get("spine_tilt_deg", 0.0),
                data.get("posture_score", 0.0),
                json.dumps(data.get("joint_velocities", [])),
                json.dumps(data.get("fft_harmonics", [])),
                json.dumps(data.get("gait_wave", [])),
                json.dumps(data.get("gait_embedding", [])),
                1 if data.get("is_valid_gait", True) else 0,
                1 if data.get("gait_usable", True) else 0
            ))
            conn.commit()
            return True
        finally:
            conn.close()

    def get_gait_dynamics(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve gait dynamics record for a track."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM gait_dynamics WHERE track_id = ?", (track_id,))
            r = cursor.fetchone()
            if not r:
                return None
            return {
                "track_id": r["track_id"],
                "person_id": r["person_id"],
                "sequence_length": r["sequence_length"],
                "stride_length_px": r["stride_length_px"],
                "stride_length_cm": r["stride_length_cm"],
                "cadence_hz": r["cadence_hz"],
                "spine_tilt_deg": r["spine_tilt_deg"],
                "posture_score": r["posture_score"],
                "joint_velocities": json.loads(r["joint_velocities_json"]) if r["joint_velocities_json"] else [],
                "fft_harmonics": json.loads(r["fft_harmonics_json"]) if r["fft_harmonics_json"] else [],
                "gait_wave": json.loads(r["gait_wave_json"]) if r["gait_wave_json"] else [],
                "gait_embedding": json.loads(r["gait_embedding_json"]) if r["gait_embedding_json"] else [],
                "is_valid_gait": bool(r["is_valid_gait"]),
                "gait_usable": bool(r["gait_usable"])
            }
        finally:
            conn.close()

    def save_enhanced_video_artifact(
        self,
        source_video_path: str,
        enhanced_video_path: str,
        track_id: Optional[str] = None,
        person_id: Optional[str] = None,
        frame_count: int = 0,
        enhancement_profile: str = "TIER_1_CLAHE_BILATERAL",
        raw_sha256: str = "",
        enhanced_sha256: str = ""
    ) -> bool:
        """Record forensic video enhancement artifact with SHA-256 integrity hash."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT INTO enhanced_video_artifacts (
                source_video_path, enhanced_video_path, track_id, person_id,
                frame_count, enhancement_profile, raw_sha256, enhanced_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                source_video_path, enhanced_video_path, track_id, person_id,
                frame_count, enhancement_profile, raw_sha256, enhanced_sha256
            ))
            conn.commit()
            return True
        finally:
            conn.close()

    def export_training_dataset(self, target_id: Optional[str] = None, track_id: Optional[str] = None) -> Dict[str, Any]:
        """Export SQL-persisted exo-skeleton sequences and gait labels formatted for neural model training.

        Format:
        - keypoint_sequences: [T, 17, 3] (x, y, confidence)
        - labels: identity target, cadence, stride, height, postures
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            if track_id:
                cursor.execute("SELECT * FROM person_skeletons WHERE track_id = ? ORDER BY frame_idx ASC", (track_id,))
            elif target_id:
                cursor.execute("SELECT * FROM person_skeletons WHERE person_id = ? ORDER BY frame_idx ASC", (target_id,))
            else:
                cursor.execute("SELECT * FROM person_skeletons ORDER BY track_id, frame_idx ASC")

            rows = cursor.fetchall()

            # Group by track_id
            sequences: Dict[str, List[Dict[str, Any]]] = {}
            for r in rows:
                tid = r["track_id"]
                if tid not in sequences:
                    sequences[tid] = []

                kpts_17 = json.loads(r["keypoints_coco_json"]) if r["keypoints_coco_json"] else []
                # Form [17, 3] array: x, y, conf
                frame_matrix = []
                for j in kpts_17:
                    frame_matrix.append([
                        j.get("crop_x", 0.0),
                        j.get("crop_y", 0.0),
                        j.get("confidence", 0.0)
                    ])

                sequences[tid].append({
                    "frame_idx": r["frame_idx"],
                    "timestamp": r["timestamp"],
                    "inter_ankle_dist": r["inter_ankle_dist"],
                    "spine_tilt_deg": r["spine_tilt_deg"],
                    "is_confident": bool(r["is_confident"]),
                    "keypoint_matrix": frame_matrix
                })

            # Fetch corresponding gait dynamics for each track
            dataset = []
            for tid, frames in sequences.items():
                gait = self.get_gait_dynamics(tid)
                dataset.append({
                    "track_id": tid,
                    "target_id": target_id or tid,
                    "frame_count": len(frames),
                    "temporal_keypoints_shape": [len(frames), 17, 3],
                    "frames": frames,
                    "gait_dynamics": gait
                })

            return {
                "sample_count": len(dataset),
                "total_frames": sum(d["frame_count"] for d in dataset),
                "schema": "COCO_17_KEYPOINTS_TEMPORAL [T, V=17, C=3]",
                "compatible_models": ["GaitGraph2", "GPGait", "ST-GCN", "SkeletonGait++"],
                "samples": dataset
            }
        finally:
            conn.close()

