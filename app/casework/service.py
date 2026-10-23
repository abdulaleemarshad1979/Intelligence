"""Case service: the single entry point for creating cases and recording evidence and decisions."""

import json
import os
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional

from app.casework import cameras as cam_inv
from app.casework import face_search, imaging, video_tools
from app.casework import media_store as ms
from app.casework.schema import connect, init_casework_db
from app.compliance.audit import AuditLogger
from app.compliance.rbac import OfficerIdentity

DECISIONS = ("ACCEPT", "REJECT", "DEFER")
BOX_KEYS = {"x", "y", "w", "h"}


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class CaseService:
    def __init__(self, db_path: Optional[str] = None, media_root: Optional[str] = None,
                 matcher: Optional[face_search.Matcher] = None, frame_source_factory=None,
                 live_interval_s: float = 2.0):
        self._matcher = matcher
        self._frame_source_factory = frame_source_factory
        self._live_interval_s = live_interval_s
        self._runners: Dict[str, face_search.LiveJobRunner] = {}
        self.db_path = init_casework_db(db_path)
        self.store = ms.MediaStore(media_root)
        self.audit = AuditLogger(self.db_path)

    def _conn(self) -> sqlite3.Connection:
        return connect(self.db_path)

    def _log(self, officer: OfficerIdentity, action: str, resource: str, details: Dict[str, Any]) -> None:
        self.audit.log_action(action, resource, details, officer)

    # ------------------------------------------------------------ cases
    def create_case(self, officer: OfficerIdentity, case_number: str, title: str, scenario: str,
                    notes: str = "") -> str:
        cid = _id("CASE")
        with self._conn() as c:
            c.execute("INSERT INTO cw_cases (case_id, case_number, title, scenario, notes, created_by, created_at)"
                      " VALUES (?,?,?,?,?,?,?)",
                      (cid, case_number, title, scenario, notes, officer.badge_number, time.time()))
        self._log(officer, "CASE_CREATED", cid, {"case_number": case_number, "scenario": scenario})
        return cid

    def get_case(self, case_id: str) -> Dict[str, Any]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM cw_cases WHERE case_id=?", (case_id,)).fetchone()
        if not row:
            raise KeyError(case_id)
        return dict(row)

    # ------------------------------------------------------------ media
    def add_original(self, officer: OfficerIdentity, case_id: str, file_path: str, *,
                     source_description: str, last_known_time: Optional[float] = None,
                     last_known_location: Optional[str] = None,
                     last_known_latitude: Optional[float] = None,
                     last_known_longitude: Optional[float] = None,
                     source_camera_id: Optional[str] = None,
                     make_proxy: bool = True) -> str:
        """Preserve the file byte-for-byte and record where it came from. Never altered afterwards."""
        self.get_case(case_id)
        filename = os.path.basename(file_path)
        kind = ms.detect_kind(filename)
        stored = self.store.put(file_path, "originals", filename)
        facts = ms.probe_video(stored.path) if kind == "VIDEO" else ms.probe_image(stored.path)
        media_id = _id("MED")
        with self._conn() as c:
            c.execute(
                "INSERT INTO cw_media (media_id, case_id, kind, role, sha256, size_bytes, storage_path, "
                "original_filename, mime_type, width, height, duration_s, nominal_fps, is_vfr, label, "
                "source_description, last_known_time, last_known_location, last_known_latitude, "
                "last_known_longitude, source_camera_id, added_by, added_at) "
                "VALUES (?,?,?,'ORIGINAL',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (media_id, case_id, kind, stored.sha256, stored.size_bytes, stored.path, filename,
                 stored.mime_type, facts.get("width"), facts.get("height"), facts.get("duration_s"),
                 facts.get("nominal_fps"), int(facts["is_vfr"]) if "is_vfr" in facts else None,
                 "ORIGINAL", source_description, last_known_time, last_known_location,
                 last_known_latitude, last_known_longitude, source_camera_id,
                 officer.badge_number, time.time()))
            if kind == "VIDEO":
                c.execute("INSERT INTO cw_frame_times (media_id, pts_seconds) VALUES (?,?)",
                          (media_id, json.dumps(facts["pts_seconds"])))
        self._log(officer, "MEDIA_ORIGINAL_ADDED", media_id,
                  {"case_id": case_id, "sha256": stored.sha256, "filename": filename})
        if kind == "VIDEO" and make_proxy:
            self.add_viewing_proxy(officer, media_id)
        return media_id

    def add_viewing_proxy(self, officer: OfficerIdentity, original_id: str) -> str:
        orig = self.get_media(original_id)
        if orig["role"] != "ORIGINAL" or orig["kind"] != "VIDEO":
            raise ValueError("Proxies are made from original videos")
        self.verify_media(original_id)
        tmp = self.store.reserve("proxy", ".mp4")
        try:
            ms.make_viewing_proxy(orig["storage_path"], tmp)
            stored = self.store.put(tmp, "proxies", "proxy.mp4")
            facts = ms.probe_video(stored.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        pid = _id("MED")
        with self._conn() as c:
            c.execute(
                "INSERT INTO cw_media (media_id, case_id, kind, role, parent_media_id, sha256, size_bytes, "
                "storage_path, original_filename, mime_type, width, height, duration_s, nominal_fps, is_vfr, "
                "label, source_description, added_by, added_at) VALUES (?,?,'VIDEO','PROXY',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (pid, orig["case_id"], original_id, stored.sha256, stored.size_bytes, stored.path,
                 orig["original_filename"], stored.mime_type, facts["width"], facts["height"],
                 facts["duration_s"], facts["nominal_fps"], int(facts["is_vfr"]),
                 "VIEWING PROXY - not evidence", f"Viewing proxy of {original_id}",
                 officer.badge_number, time.time()))
        self._log(officer, "MEDIA_PROXY_CREATED", pid, {"original": original_id})
        return pid

    def add_derived(self, officer: OfficerIdentity, parent_id: str, result_path: str, *,
                    operation: str, parameters: Optional[Dict[str, Any]] = None, tool: str = "",
                    result_kind: Optional[str] = None) -> str:
        """Register an edited/enhanced copy. The parent is untouched; the step is logged and labelled."""
        parent = self.get_media(parent_id)
        if parent["role"] == "PROXY":
            raise ValueError("Derive from the original or another derived copy, not a proxy")
        self.verify_media(parent_id)
        stored = self.store.put(result_path, "derived", os.path.basename(result_path))
        kind = result_kind or parent["kind"]
        facts = ms.probe_video(stored.path) if kind == "VIDEO" else ms.probe_image(stored.path)
        mid = _id("MED")
        with self._conn() as c:
            chain = self._chain_len(c, parent_id) + 1
            c.execute(
                "INSERT INTO cw_media (media_id, case_id, kind, role, parent_media_id, sha256, size_bytes, "
                "storage_path, original_filename, mime_type, width, height, duration_s, nominal_fps, is_vfr, "
                "label, source_description, added_by, added_at) VALUES (?,?,?,'DERIVED',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (mid, parent["case_id"], kind, parent_id, stored.sha256, stored.size_bytes,
                 stored.path, os.path.basename(result_path), stored.mime_type, facts.get("width"),
                 facts.get("height"), facts.get("duration_s"), facts.get("nominal_fps"),
                 int(facts["is_vfr"]) if "is_vfr" in facts else None,
                 f"DERIVED COPY - {operation}", f"{operation} applied to {parent_id}",
                 officer.badge_number, time.time()))
            c.execute(
                "INSERT INTO cw_derivations (derivation_id, case_id, parent_media_id, result_media_id, "
                "step_index, operation, parameters, tool, performed_by, performed_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (_id("DER"), parent["case_id"], parent_id, mid, chain, operation,
                 json.dumps(parameters or {}, sort_keys=True), tool, officer.badge_number, time.time()))
            if kind == "VIDEO":
                c.execute("INSERT INTO cw_frame_times (media_id, pts_seconds) VALUES (?,?)",
                          (mid, json.dumps(facts["pts_seconds"])))
        self._log(officer, "MEDIA_DERIVED_ADDED", mid,
                  {"parent": parent_id, "operation": operation, "sha256": stored.sha256})
        return mid

    @staticmethod
    def _chain_len(c: sqlite3.Connection, media_id: str) -> int:
        n, cur = 0, media_id
        while True:
            row = c.execute("SELECT parent_media_id FROM cw_media WHERE media_id=?", (cur,)).fetchone()
            if not row or not row[0]:
                return n
            n, cur = n + 1, row[0]

    def get_media(self, media_id: str) -> Dict[str, Any]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM cw_media WHERE media_id=?", (media_id,)).fetchone()
        if not row:
            raise KeyError(media_id)
        return dict(row)

    def lineage(self, media_id: str) -> List[Dict[str, Any]]:
        """Original -> ... -> this copy, with the logged operation at each step."""
        with self._conn() as c:
            chain, cur = [], media_id
            while cur:
                m = dict(c.execute("SELECT * FROM cw_media WHERE media_id=?", (cur,)).fetchone())
                d = c.execute("SELECT operation, parameters, tool, performed_by, performed_at "
                              "FROM cw_derivations WHERE result_media_id=?", (cur,)).fetchone()
                chain.append({"media": m, "derivation": dict(d) if d else None})
                cur = m["parent_media_id"]
        return chain[::-1]

    def verify_media(self, media_id: str) -> None:
        m = self.get_media(media_id)
        self.store.verify(m["storage_path"], m["sha256"])

    # ------------------------------------------------------------ video timing
    def frame_times(self, media_id: str) -> List[float]:
        with self._conn() as c:
            row = c.execute("SELECT pts_seconds FROM cw_frame_times WHERE media_id=?", (media_id,)).fetchone()
        if not row:
            raise KeyError(f"No frame index for {media_id}")
        return json.loads(row[0])

    def frame_time(self, media_id: str, frame_no: int) -> float:
        return self.frame_times(media_id)[frame_no]

    def frame_at(self, media_id: str, t: float) -> int:
        """Index of the frame on screen at time t (last frame with pts <= t)."""
        import bisect
        times = self.frame_times(media_id)
        return max(0, bisect.bisect_right(times, t) - 1)

    # ------------------------------------------------------------ quality
    def assess_image(self, officer: OfficerIdentity, media_id: str) -> Dict[str, Any]:
        m = self.get_media(media_id)
        if m["kind"] != "IMAGE":
            raise ValueError("Quality assessment applies to images")
        res = ms.assess_image_quality(m["storage_path"])
        self._add_note(media_id, "AUTO", res["metrics"], res["warnings"], "", officer.badge_number)
        return res

    def add_quality_note(self, officer: OfficerIdentity, media_id: str, note: str,
                         warnings: Optional[List[str]] = None) -> None:
        self.get_media(media_id)
        self._add_note(media_id, "MANUAL", {}, warnings or [], note, officer.badge_number)

    def _add_note(self, media_id, origin, metrics, warnings, note, author) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO cw_quality_notes (note_id, media_id, origin, metrics, warnings, note, "
                      "author, created_at) VALUES (?,?,?,?,?,?,?,?)",
                      (_id("QN"), media_id, origin, json.dumps(metrics), json.dumps(warnings), note,
                       author, time.time()))

    # ------------------------------------------------------------ annotations
    def annotate(self, officer: OfficerIdentity, media_id: str, kind: str, *, start_s: Optional[float] = None,
                 end_s: Optional[float] = None, bbox: Optional[Dict[str, float]] = None,
                 label: str = "", note: str = "") -> str:
        """Interval/person/region/note markers. They sit beside the media; the file is never edited."""
        m = self.get_media(media_id)
        if bbox is not None and set(bbox) != BOX_KEYS:
            raise ValueError("bbox needs normalised x, y, w, h")
        if kind in ("INTERVAL",) and (start_s is None or end_s is None):
            raise ValueError("INTERVAL needs start_s and end_s")
        if m["duration_s"] and end_s is not None and end_s > m["duration_s"] + 0.5:
            raise ValueError("end_s is beyond the end of the video")
        aid = _id("ANN")
        with self._conn() as c:
            c.execute("INSERT INTO cw_annotations (annotation_id, case_id, media_id, kind, start_s, end_s, "
                      "bbox, label, note, author, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (aid, m["case_id"], media_id, kind, start_s, end_s,
                       json.dumps(bbox) if bbox else None, label, note, officer.badge_number, time.time()))
        return aid

    def annotations(self, media_id: str, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        q, args = "SELECT * FROM cw_annotations WHERE media_id=?", [media_id]
        if kind:
            q, args = q + " AND kind=?", args + [kind]
        with self._conn() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY start_s, created_at", args)]

    # ------------------------------------------------------------ cameras / clips
    def import_cameras(self, xlsx_path: str) -> Dict[str, Any]:
        with self._conn() as c:
            return cam_inv.import_camera_sheet(c, xlsx_path)

    def propose_candidates(self, officer: OfficerIdentity, case_id: str, media_id: str, *,
                           radius_m: float = 500.0, window_before_s: float = 600.0,
                           window_after_s: float = 1800.0,
                           camera_types: Optional[List[str]] = None) -> List[str]:
        """Case-scoped retrieval by place and time only: cameras near the reference's last known
        location, over a window around its last known time. No image matching is performed."""
        m = self.get_media(media_id)
        if m["case_id"] != case_id:
            raise ValueError("Media belongs to a different case")
        if None in (m["last_known_time"], m["last_known_latitude"], m["last_known_longitude"]):
            raise ValueError("Reference needs last-known time and coordinates to propose cameras")
        with self._conn() as c:
            near = cam_inv.cameras_near(c, m["last_known_latitude"], m["last_known_longitude"],
                                        radius_m, camera_types)
            ids = []
            for cam in near:
                cid = _id("CLIP")
                try:
                    c.execute("INSERT INTO cw_clip_candidates (candidate_id, case_id, camera_id, window_start, "
                              "window_end, origin, rationale, created_by, created_at) "
                              "VALUES (?,?,?,?,?,'SPATIOTEMPORAL',?,?,?)",
                              (cid, case_id, cam["camera_id"], m["last_known_time"] - window_before_s,
                               m["last_known_time"] + window_after_s,
                               f"{cam['distance_m']} m from last known location", officer.badge_number, time.time()))
                    ids.append(cid)
                except sqlite3.IntegrityError:
                    continue
        self._log(officer, "CLIP_CANDIDATES_PROPOSED", case_id, {"count": len(ids), "radius_m": radius_m})
        return ids

    def add_manual_candidate(self, officer: OfficerIdentity, case_id: str, camera_id: str,
                             window_start: float, window_end: float, rationale: str,
                             media_id: Optional[str] = None) -> str:
        cid = _id("CLIP")
        with self._conn() as c:
            c.execute("INSERT INTO cw_clip_candidates (candidate_id, case_id, camera_id, media_id, window_start, "
                      "window_end, origin, rationale, created_by, created_at) VALUES (?,?,?,?,?,?,'MANUAL',?,?,?)",
                      (cid, case_id, camera_id, media_id, window_start, window_end, rationale,
                       officer.badge_number, time.time()))
        return cid

    # ------------------------------------------------------------ review queue
    def record_decision(self, officer: OfficerIdentity, candidate_id: str, decision: str, reason: str = "") -> str:
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        if decision == "REJECT" and not reason.strip():
            raise ValueError("A rejection needs a reason")
        with self._conn() as c:
            cand = c.execute("SELECT case_id FROM cw_clip_candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
            if not cand:
                raise KeyError(candidate_id)
            did = _id("DEC")
            c.execute("INSERT INTO cw_review_decisions (decision_id, case_id, candidate_id, decision, "
                      "reviewer_badge, reviewer_name, reason, decided_at) VALUES (?,?,?,?,?,?,?,?)",
                      (did, cand["case_id"], candidate_id, decision, officer.badge_number,
                       officer.full_name, reason, time.time()))
        self._log(officer, "CLIP_REVIEW_DECISION", candidate_id, {"decision": decision, "reason": reason})
        return did

    def review_queue(self, case_id: str, status: Optional[str] = None, limit: Optional[int] = None,
                     offset: int = 0, order: str = "time") -> List[Dict[str, Any]]:
        """Candidates with their latest decision (PENDING if none). History stays in cw_review_decisions."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT k.*, cam.location AS camera_location, cam.camera_type, "
                " COALESCE((SELECT decision FROM cw_review_decisions d WHERE d.candidate_id=k.candidate_id "
                "           ORDER BY decided_at DESC, rowid DESC LIMIT 1), 'PENDING') AS status "
                "FROM cw_clip_candidates k JOIN cw_cameras cam USING (camera_id) WHERE k.case_id=? "
                "ORDER BY k.window_start, k.camera_id", (case_id,)).fetchall()
        out = [dict(r) for r in rows]
        if status:
            out = [r for r in out if r["status"] == status]
        if order == "score":
            out.sort(key=lambda r: -(r["score"] or 0))
        return out[offset:offset + limit] if limit is not None else out[offset:]

    def decision_history(self, candidate_id: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM cw_review_decisions WHERE candidate_id=? ORDER BY decided_at, rowid", (candidate_id,))]

    # ------------------------------------------------------------ derived-copy tools
    def _derive(self, officer, parent_id, operation, suffix, make, result_kind=None, parameters=None, tool=""):
        """Run make(src_path, dst_path) -> params on the verified parent and register the result."""
        parent = self.get_media(parent_id)
        self.verify_media(parent_id)
        tmp = self.store.reserve("derived", suffix)
        try:
            params = make(parent["storage_path"], tmp) or {}
            params.update(parameters or {})
            return self.add_derived(officer, parent_id, tmp, operation=operation, parameters=params,
                                    tool=tool, result_kind=result_kind)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def _suffix(self, media_id: str) -> str:
        return os.path.splitext(self.get_media(media_id)["original_filename"])[1] or ".png"

    def crop_image(self, officer, media_id: str, x: int, y: int, w: int, h: int) -> str:
        if self.get_media(media_id)["kind"] != "IMAGE":
            raise ValueError("Crop applies to images; extract a frame first")
        return self._derive(officer, media_id, "CROP", ".png",
                            lambda s, d: imaging.crop(s, d, x, y, w, h), tool="opencv")

    def enhance_image(self, officer, media_id: str, operation: str, **params) -> str:
        if self.get_media(media_id)["kind"] != "IMAGE":
            raise ValueError("Enhancement applies to images; extract a frame first")
        return self._derive(officer, media_id, operation, ".png",
                            lambda s, d: imaging.enhance(s, d, operation, **params), tool="opencv")

    def extract_frame(self, officer, video_id: str, frame_no: int) -> str:
        m = self.get_media(video_id)
        if m["kind"] != "VIDEO" or m["role"] == "PROXY":
            raise ValueError("Extract frames from an original or derived video, not a proxy")
        pts = self.frame_times(video_id)
        return self._derive(officer, video_id, "FRAME_EXTRACT", ".png",
                            lambda s, d: video_tools.extract_frame(s, d, frame_no, pts),
                            result_kind="IMAGE", tool="opencv")

    def contact_sheet(self, officer, video_id: str, start_s: float, end_s: float, **kw) -> str:
        pts = self.frame_times(video_id)
        return self._derive(officer, video_id, "CONTACT_SHEET", ".png",
                            lambda s, d: video_tools.contact_sheet(s, d, pts, start_s, end_s, **kw),
                            result_kind="IMAGE", tool="opencv")

    def slow_motion(self, officer, video_id: str, start_s: float, end_s: float, speed: float = 0.25) -> str:
        m = self.get_media(video_id)
        if m["role"] == "PROXY":
            raise ValueError("Use the original or a derived video")
        return self._derive(officer, video_id, "SLOW_MOTION", ".mp4",
                            lambda s, d: video_tools.slow_motion(s, d, start_s, end_s, speed), tool="ffmpeg")

    def compare_images(self, officer, media_a: str, media_b: str, label_a: str = "", label_b: str = "") -> str:
        """Side-by-side sheet for manual comparison; recorded as a derived copy of A that names B."""
        a, b = self.get_media(media_a), self.get_media(media_b)
        if a["case_id"] != b["case_id"] or "IMAGE" not in (a["kind"], b["kind"]) or a["kind"] != b["kind"]:
            raise ValueError("Compare two images from the same case")
        self.verify_media(media_b)
        return self._derive(
            officer, media_a, "SIDE_BY_SIDE", ".png",
            lambda s, d: imaging.side_by_side(s, b["storage_path"], d, label_a or a["label"], label_b or b["label"]),
            parameters={"other_media_id": media_b, "other_sha256": b["sha256"]}, tool="opencv")

    def compare_frames(self, officer, video_a: str, frame_a: int, video_b: str, frame_b: int) -> str:
        a, b = self.get_media(video_a), self.get_media(video_b)
        if a["case_id"] != b["case_id"] or a["kind"] != "VIDEO" or b["kind"] != "VIDEO":
            raise ValueError("Compare frames of two videos from the same case")
        self.verify_media(video_b)
        la = f"#{frame_a} @{self.frame_time(video_a, frame_a):.2f}s {a['original_filename']}"
        lb = f"#{frame_b} @{self.frame_time(video_b, frame_b):.2f}s {b['original_filename']}"
        return self._derive(
            officer, video_a, "FRAME_COMPARISON", ".png",
            lambda s, d: video_tools.paired_frames(s, frame_a, b["storage_path"], frame_b, d, la, lb),
            result_kind="IMAGE", parameters={"other_media_id": video_b, "other_sha256": b["sha256"],
                                            "frame_b": frame_b}, tool="opencv")

    def compare_videos(self, officer, video_a: str, video_b: str, speed: float = 1.0) -> str:
        a, b = self.get_media(video_a), self.get_media(video_b)
        if a["case_id"] != b["case_id"] or a["kind"] != "VIDEO" or b["kind"] != "VIDEO":
            raise ValueError("Compare two videos from the same case")
        self.verify_media(video_b)
        return self._derive(
            officer, video_a, "PAIRED_PLAYBACK", ".mp4",
            lambda s, d: video_tools.paired_playback(s, b["storage_path"], d, speed),
            parameters={"other_media_id": video_b, "other_sha256": b["sha256"]}, tool="ffmpeg")

    # ------------------------------------------------------------ descriptors
    def add_descriptor(self, officer, case_id: str, subject_label: str, category: str, value: str, *,
                       basis: str = "OBSERVED", confidence: str = "MEDIUM", media_id: Optional[str] = None,
                       at_time_s: Optional[float] = None, note: str = "") -> str:
        """A human-recorded characteristic. Height needs a stated measurement reference."""
        self.get_case(case_id)
        if not value.strip():
            raise ValueError("A descriptor needs a value")
        if category == "HEIGHT_ESTIMATE" and not note.strip():
            raise ValueError("Height estimates must state their reference in note (e.g. 'door frame 2.1 m')")
        did = _id("DSC")
        with self._conn() as c:
            c.execute("INSERT INTO cw_descriptors (descriptor_id, case_id, subject_label, media_id, category, "
                      "value, basis, confidence, at_time_s, note, author, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                      (did, case_id, subject_label, media_id, category, value.strip(), basis, confidence,
                       at_time_s, note, officer.badge_number, time.time()))
        self._log(officer, "DESCRIPTOR_ADDED", did, {"case_id": case_id, "category": category})
        return did

    def retract_descriptor(self, officer, descriptor_id: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE cw_descriptors SET status='RETRACTED' WHERE descriptor_id=?", (descriptor_id,))
        self._log(officer, "DESCRIPTOR_RETRACTED", descriptor_id, {})

    def descriptors(self, case_id: str, subject_label: Optional[str] = None,
                    include_retracted: bool = False) -> List[Dict[str, Any]]:
        q, a = "SELECT * FROM cw_descriptors WHERE case_id=?", [case_id]
        if subject_label:
            q, a = q + " AND subject_label=?", a + [subject_label]
        if not include_retracted:
            q += " AND status='ACTIVE'"
        with self._conn() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY subject_label, category, created_at", a)]

    # ------------------------------------------------------------ face-similarity search (leads only)
    @property
    def matcher(self) -> face_search.Matcher:
        if self._matcher is None:
            self._matcher = face_search.SFaceMatcher()
        return self._matcher

    def create_search_job(self, officer: OfficerIdentity, case_id: str, *, mode: str,
                          reference_media_id: str, authority_ref: str,
                          camera_ids: Optional[List[str]] = None, clip_media_ids: Optional[List[str]] = None,
                          threshold: float = face_search.DEFAULT_THRESHOLD,
                          duration_hours: float = face_search.DEFAULT_LIVE_HOURS) -> str:
        """Register a bounded search. Needs a stated authority, one reference face from this case, and
        explicit cameras (LIVE, capped, expiring) or explicit footage (RECORDED). Nothing starts yet."""
        if mode not in ("RECORDED", "LIVE"):
            raise ValueError("mode must be RECORDED or LIVE")
        self.get_case(case_id)
        if len(authority_ref.strip()) < 4:
            raise ValueError("authority_ref (FIR / court or supervisory order reference) is required")
        if threshold < face_search.MIN_THRESHOLD:
            raise ValueError(f"threshold may not be below {face_search.MIN_THRESHOLD}")
        ref = self.get_media(reference_media_id)
        if ref["case_id"] != case_id or ref["kind"] != "IMAGE" or ref["role"] == "PROXY":
            raise ValueError("Reference must be an image belonging to this case")
        self.verify_media(reference_media_id)
        face_search.reference_embedding(self.matcher, ref["storage_path"])   # exactly one face, or ValueError
        camera_ids, clip_media_ids = list(dict.fromkeys(camera_ids or [])), list(dict.fromkeys(clip_media_ids or []))
        expires = None
        with self._conn() as c:
            if mode == "LIVE":
                if not camera_ids:
                    raise ValueError("Live search needs an explicit camera list")
                if len(camera_ids) > face_search.MAX_LIVE_CAMERAS:
                    raise ValueError(f"Live search is limited to {face_search.MAX_LIVE_CAMERAS} cameras per job")
                if not 0 < duration_hours <= face_search.MAX_LIVE_HOURS:
                    raise ValueError(f"duration_hours must be in (0, {face_search.MAX_LIVE_HOURS}]")
                known = {r[0] for r in c.execute(
                    f"SELECT camera_id FROM cw_cameras WHERE camera_id IN ({','.join('?' * len(camera_ids))})", camera_ids)}
                if known != set(camera_ids):
                    raise ValueError(f"Unknown camera(s): {sorted(set(camera_ids) - known)}")
            else:
                if not clip_media_ids:
                    raise ValueError("Recorded search needs the footage (clip_media_ids) to search")
                for mid in clip_media_ids:
                    m = self.get_media(mid)
                    if m["case_id"] != case_id or m["kind"] != "VIDEO" or m["role"] == "PROXY":
                        raise ValueError(f"{mid} is not an original/derived video in this case")
            job_id = _id("JOB")
            c.execute("INSERT INTO cw_search_jobs (job_id, case_id, mode, reference_media_id, authority_ref, camera_ids, "
                      "clip_media_ids, threshold, duration_hours, created_by, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (job_id, case_id, mode, reference_media_id, authority_ref.strip(), json.dumps(camera_ids),
                       json.dumps(clip_media_ids), threshold,
                       duration_hours if mode == "LIVE" else None, officer.badge_number, time.time()))
        self._log(officer, "FACE_SEARCH_JOB_CREATED", job_id,
                  {"case_id": case_id, "mode": mode, "authority_ref": authority_ref.strip(),
                   "cameras": camera_ids, "clips": clip_media_ids, "threshold": threshold,
                   "duration_hours": duration_hours if mode == "LIVE" else None})
        return job_id

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM cw_search_jobs WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            raise KeyError(job_id)
        d = dict(row)
        d["camera_ids"], d["clip_media_ids"] = json.loads(d["camera_ids"]), json.loads(d["clip_media_ids"])
        return d

    def list_jobs(self, case_id: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            ids = [r[0] for r in c.execute("SELECT job_id FROM cw_search_jobs WHERE case_id=? ORDER BY created_at DESC", (case_id,))]
        return [self.get_job(i) for i in ids]

    def _job_update(self, job_id: str, **cols) -> None:
        sets = ", ".join(f"{k}=?" for k in cols)
        with self._conn() as c:
            c.execute(f"UPDATE cw_search_jobs SET {sets} WHERE job_id=?", list(cols.values()) + [job_id])

    def _job_cap_reached(self, job_id: str) -> bool:
        return self.get_job(job_id)["candidates_found"] >= face_search.MAX_CANDIDATES_PER_JOB

    def _add_face_candidate(self, job: Dict[str, Any], camera_id: str, media_id: str, t0: float, t1: float,
                            frame_no: Optional[int], time_s: Optional[float], score: float,
                            snapshot_id: Optional[str]) -> Optional[str]:
        cid = _id("CLIP")
        with self._conn() as c:
            try:
                c.execute("INSERT INTO cw_clip_candidates (candidate_id, case_id, camera_id, media_id, window_start, "
                          "window_end, origin, job_id, frame_no, frame_time_s, score, snapshot_media_id, rationale, "
                          "created_by, created_at) VALUES (?,?,?,?,?,?,'FACE_SEARCH',?,?,?,?,?,?,?,?)",
                          (cid, job["case_id"], camera_id, media_id, t0, t1, job["job_id"], frame_no, time_s,
                           round(score, 4), snapshot_id,
                           "Face similarity above threshold (unvalidated score). Requires human review.",
                           job["created_by"], time.time()))
            except sqlite3.IntegrityError:
                return None
            c.execute("UPDATE cw_search_jobs SET candidates_found = candidates_found + 1 WHERE job_id=?", (job["job_id"],))
        return cid

    def run_recorded_search(self, officer: OfficerIdentity, job_id: str,
                            should_stop=lambda: False) -> Dict[str, Any]:
        """Scan the job's footage; every hit becomes a PENDING review candidate with a snapshot frame."""
        job = self.get_job(job_id)
        if job["mode"] != "RECORDED" or job["status"] != "CREATED":
            raise ValueError("Job is not a startable recorded search")
        self._job_update(job_id, status="RUNNING")
        self._log(officer, "FACE_SEARCH_STARTED", job_id, {"mode": "RECORDED"})
        status, err = "COMPLETED", None
        try:
            ref = self.get_media(job["reference_media_id"])
            self.verify_media(ref["media_id"])
            ref_emb = face_search.reference_embedding(self.matcher, ref["storage_path"])
            for mid in job["clip_media_ids"]:
                m = self.get_media(mid)
                self.verify_media(mid)
                pts = self.frame_times(mid)
                cam = m["source_camera_id"]
                if not cam:
                    raise ValueError(f"{mid} has no source camera; set source_camera_id when adding it")
                base = m["last_known_time"] or 0.0
                hits = face_search.scan_video(
                    self.matcher, ref_emb, m["storage_path"], pts, job["threshold"], should_stop,
                    lambda n: self._job_update(job_id, frames_scanned=self.get_job(job_id)["frames_scanned"] + n))
                for h in hits:
                    if self._job_cap_reached(job_id):
                        status = "CAPPED"
                        break
                    snap = self.extract_frame(officer, mid, h["frame_no"])
                    self._add_face_candidate(job, cam, mid, base + h["time_s"] - 1.0, base + h["time_s"] + 1.0,
                                             h["frame_no"], h["time_s"], h["score"], snap)
                if status == "CAPPED" or should_stop():
                    status = "CAPPED" if status == "CAPPED" else "STOPPED"
                    break
        except Exception as e:
            status, err = "FAILED", str(e)
        self._job_update(job_id, status=status, error=err, finished_at=time.time())
        self._log(officer, "FACE_SEARCH_FINISHED", job_id, {"status": status, "error": err})
        return self.get_job(job_id)

    def start_live_search(self, officer: OfficerIdentity, job_id: str) -> None:
        job = self.get_job(job_id)
        if job["mode"] != "LIVE" or job["status"] != "CREATED":
            raise ValueError("Job is not a startable live search")
        if self._frame_source_factory is None:
            raise ValueError("No live frame source is configured on this server")
        job["expires_at"] = time.time() + (job["duration_hours"] or face_search.DEFAULT_LIVE_HOURS) * 3600
        self._job_update(job_id, status="RUNNING", expires_at=job["expires_at"])
        ref = self.get_media(job["reference_media_id"])
        self.verify_media(ref["media_id"])
        ref_emb = face_search.reference_embedding(self.matcher, ref["storage_path"])

        def on_hit(cam, frame, score, box):
            import cv2
            import tempfile
            now = time.time()
            fd, tmp = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            try:
                cv2.imwrite(tmp, frame)
                snap = self.add_original(officer, job["case_id"], tmp,
                                         source_description=f"Live capture from {cam} for search {job_id}",
                                         last_known_time=now, source_camera_id=cam)
            finally:
                os.remove(tmp)
            self._add_face_candidate(job, cam, snap, now - 1.0, now + 1.0, None, None, score, snap)

        def on_tick(n):
            self._job_update(job_id, frames_scanned=self.get_job(job_id)["frames_scanned"] + n)

        def on_end(status, err):
            self._runners.pop(job_id, None)
            self._job_update(job_id, status=status, error=err, finished_at=time.time())
            self._log(officer, "FACE_SEARCH_FINISHED", job_id, {"status": status, "error": err})

        runner = face_search.LiveJobRunner(job, ref_emb, self.matcher, self._frame_source_factory(),
                                           on_hit, on_tick, on_end, lambda: self._job_cap_reached(job_id),
                                           self._live_interval_s)
        self._runners[job_id] = runner
        self._log(officer, "FACE_SEARCH_STARTED", job_id, {"mode": "LIVE", "expires_at": job["expires_at"]})
        runner.start()

    def stop_search(self, officer: OfficerIdentity, job_id: str) -> None:
        r = self._runners.get(job_id)
        if r:
            r.stop()
            r.join(timeout=15)
        else:
            job = self.get_job(job_id)
            if job["status"] in ("CREATED", "RUNNING"):
                self._job_update(job_id, status="STOPPED", finished_at=time.time())
        self._log(officer, "FACE_SEARCH_STOPPED", job_id, {})

    def resume_after_restart(self) -> int:
        """Live jobs do not survive a restart; mark them stopped rather than leave them 'RUNNING'."""
        with self._conn() as c:
            return c.execute("UPDATE cw_search_jobs SET status='STOPPED', finished_at=? WHERE status='RUNNING'",
                             (time.time(),)).rowcount
