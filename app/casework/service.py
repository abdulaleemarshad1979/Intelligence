"""Case service: the single entry point for creating cases and recording evidence and decisions."""

import json
import os
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional

from app.casework import cameras as cam_inv
from app.casework import media_store as ms
from app.casework.schema import connect, init_casework_db
from app.compliance.audit import AuditLogger
from app.compliance.rbac import OfficerIdentity

DECISIONS = ("ACCEPT", "REJECT", "DEFER")
BOX_KEYS = {"x", "y", "w", "h"}


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class CaseService:
    def __init__(self, db_path: Optional[str] = None, media_root: Optional[str] = None):
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
                    operation: str, parameters: Optional[Dict[str, Any]] = None, tool: str = "") -> str:
        """Register an edited/enhanced copy. The parent is untouched; the step is logged and labelled."""
        parent = self.get_media(parent_id)
        if parent["role"] == "PROXY":
            raise ValueError("Derive from the original or another derived copy, not a proxy")
        self.verify_media(parent_id)
        stored = self.store.put(result_path, "derived", os.path.basename(result_path))
        facts = ms.probe_video(stored.path) if parent["kind"] == "VIDEO" else ms.probe_image(stored.path)
        mid = _id("MED")
        with self._conn() as c:
            chain = self._chain_len(c, parent_id) + 1
            c.execute(
                "INSERT INTO cw_media (media_id, case_id, kind, role, parent_media_id, sha256, size_bytes, "
                "storage_path, original_filename, mime_type, width, height, duration_s, nominal_fps, is_vfr, "
                "label, source_description, added_by, added_at) VALUES (?,?,?,'DERIVED',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (mid, parent["case_id"], parent["kind"], parent_id, stored.sha256, stored.size_bytes,
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
            if parent["kind"] == "VIDEO":
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

    def review_queue(self, case_id: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Candidates with their latest decision (PENDING if none). History stays in cw_review_decisions."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT k.*, cam.location AS camera_location, cam.camera_type, "
                " COALESCE((SELECT decision FROM cw_review_decisions d WHERE d.candidate_id=k.candidate_id "
                "           ORDER BY decided_at DESC, rowid DESC LIMIT 1), 'PENDING') AS status "
                "FROM cw_clip_candidates k JOIN cw_cameras cam USING (camera_id) WHERE k.case_id=? "
                "ORDER BY k.window_start, k.camera_id", (case_id,)).fetchall()
        out = [dict(r) for r in rows]
        return [r for r in out if r["status"] == status] if status else out

    def decision_history(self, candidate_id: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM cw_review_decisions WHERE candidate_id=? ORDER BY decided_at, rowid", (candidate_id,))]
