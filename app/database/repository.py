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
            cursor.execute("""
            INSERT OR REPLACE INTO criminal_records (
                id, fir_no, unit_name, subdivision, police_station, accused_name,
                alias, age, gender, acts_sec, brief_facts, latitude, longitude,
                status_of_case, photo_url, known_height_cm, torso_leg_ratio,
                stride_length_cm, posture_lean_angle, posture_correctness,
                clothing_upper_color, clothing_lower_color,
                face_embedding, body_embedding, gait_embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.id, record.fir_no, record.unit_name, record.subdivision, record.police_station,
                record.accused_name, record.alias, record.age, record.gender, record.acts_sec,
                record.brief_facts, record.latitude, record.longitude, record.status_of_case,
                record.photo_url, record.known_height_cm, record.torso_leg_ratio, record.stride_length_cm,
                record.posture_lean_angle, record.posture_correctness, record.clothing_upper_color,
                record.clothing_lower_color, json.dumps(record.face_embedding),
                json.dumps(record.body_embedding), json.dumps(record.gait_embedding)
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
                connected_topology_json, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (cam.camera_id, cam.name, cam.latitude, cam.longitude, cam.zone,
                  cam.view_direction, json.dumps(cam.connected_topology), 1 if cam.is_active else 0))
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
            return [
                CameraEntity(
                    camera_id=r["camera_id"],
                    name=r["name"],
                    latitude=r["latitude"],
                    longitude=r["longitude"],
                    zone=r["zone"] or "Central Division",
                    view_direction=r["view_direction"] or "NORTH",
                    connected_topology=json.loads(r["connected_topology_json"]) if r["connected_topology_json"] else [],
                    is_active=bool(r["is_active"])
                )
                for r in rows
            ]
        finally:
            conn.close()

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
