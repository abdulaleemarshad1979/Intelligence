"""Face-similarity search: bounded jobs, recorded and live scans, and the human review queue."""

import glob
import os
import subprocess
import time

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes_casework
from app.casework import face_search
from app.casework.schema import connect
from app.casework.service import CaseService
from app.compliance.rbac import DEFAULT_OFFICER as O

RED, BLUE, GREEN = (0, 0, 255), (255, 0, 0), (0, 255, 0)   # BGR


class FakeMatcher:
    """'Face' = any non-dark image; embedding = unit mean colour. Same colour -> similarity 1."""

    def embed_faces(self, img):
        m = img.reshape(-1, 3).mean(axis=0)
        if m.sum() < 30:
            return []
        return [(m / np.linalg.norm(m), (0, 0, img.shape[1], img.shape[0]), 1.0)]

    def similarity(self, a, b):
        return float(np.dot(a, b))


class FakeSource:
    def __init__(self, frames):
        self.frames, self.closed = frames, False

    def grab(self, camera_id):
        return self.frames.get(camera_id)

    def close(self):
        self.closed = True


def _solid(path, bgr):
    cv2.imwrite(str(path), np.full((120, 120, 3), bgr, dtype=np.uint8))
    return str(path)


def _clip(path, colours, seg_s=2, fps=10):
    parts = ";".join(f"color=c=0x{r:02x}{g:02x}{b:02x}:s=64x64:d={seg_s}:r={fps}[v{i}]"
                     for i, (b, g, r) in enumerate(colours))
    chain = "".join(f"[v{i}]" for i in range(len(colours)))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-filter_complex", f"{parts};{chain}concat=n={len(colours)}[o]",
                    "-map", "[o]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


def _cameras(svc, n=3):
    with connect(svc.db_path) as c:
        for i in range(1, n + 1):
            c.execute("INSERT INTO cw_cameras (camera_id, ip_address, camera_type, location, latitude, longitude, "
                      "imported_at) VALUES (?,?,?,?,?,?,0)", (f"CAM{i}", f"10.0.0.{i}", "FIXED", f"Spot {i}", 17.0, 81.8))


@pytest.fixture
def svc(tmp_path):
    s = CaseService(str(tmp_path / "cw.db"), str(tmp_path / "media"), matcher=FakeMatcher(),
                    frame_source_factory=lambda: FakeSource({}), live_interval_s=0.05)
    _cameras(s)
    return s


@pytest.fixture
def case_ref(svc, tmp_path):
    case = svc.create_case(O, "FS-1", "search", "PHOTO")
    ref = svc.add_original(O, case, _solid(tmp_path / "ref.png", RED), source_description="reference")
    return case, ref


def test_job_guardrails(svc, case_ref, tmp_path):
    case, ref = case_ref
    mk = lambda **kw: svc.create_search_job(O, case, **{"mode": "LIVE", "reference_media_id": ref,
                                                        "authority_ref": "FIR 123/2026", "camera_ids": ["CAM1"], **kw})
    assert mk()
    with pytest.raises(ValueError, match="authority_ref"):
        mk(authority_ref="x")
    with pytest.raises(ValueError, match="threshold"):
        mk(threshold=0.2)
    with pytest.raises(ValueError, match="explicit camera"):
        mk(camera_ids=[])
    with pytest.raises(ValueError, match="limited to"):
        mk(camera_ids=[f"C{i}" for i in range(21)])
    with pytest.raises(ValueError, match="Unknown camera"):
        mk(camera_ids=["NOPE"])
    with pytest.raises(ValueError, match="duration_hours"):
        mk(duration_hours=48)
    dark = svc.add_original(O, case, _solid(tmp_path / "dark.png", (0, 0, 0)), source_description="no face")
    with pytest.raises(ValueError, match="exactly one face"):
        mk(reference_media_id=dark)
    other = svc.create_case(O, "FS-2", "other", "PHOTO")
    with pytest.raises(ValueError, match="belonging to this case"):
        svc.create_search_job(O, other, mode="LIVE", reference_media_id=ref, authority_ref="FIR 1/2026", camera_ids=["CAM1"])
    with pytest.raises(ValueError, match="footage"):
        svc.create_search_job(O, case, mode="RECORDED", reference_media_id=ref, authority_ref="FIR 123/2026")


def test_recorded_search_yields_pending_candidates_with_snapshots(svc, case_ref, tmp_path):
    case, ref = case_ref
    clip = svc.add_original(O, case, _clip(tmp_path / "c.mp4", [RED, BLUE, RED]), source_description="cam 2 footage",
                            last_known_time=1_700_000_000.0, source_camera_id="CAM2", make_proxy=False)
    job = svc.create_search_job(O, case, mode="RECORDED", reference_media_id=ref, authority_ref="FIR 123/2026",
                                clip_media_ids=[clip])
    done = svc.run_recorded_search(O, job)
    assert done["status"] == "COMPLETED" and done["candidates_found"] == 2 and done["frames_scanned"] > 8
    q = svc.review_queue(case, "PENDING")
    assert [r["camera_id"] for r in q] == ["CAM2", "CAM2"] and all(r["origin"] == "FACE_SEARCH" for r in q)
    assert q[0]["frame_time_s"] < 2.0 < 4.0 <= q[1]["frame_time_s"] and q[0]["score"] > 0.9
    assert svc.get_media(q[0]["snapshot_media_id"])["kind"] == "IMAGE"
    svc.record_decision(O, q[0]["candidate_id"], "REJECT", "different clothing")
    svc.record_decision(O, q[1]["candidate_id"], "DEFER")
    assert {r["status"] for r in svc.review_queue(case)} == {"REJECT", "DEFER"}
    assert len(svc.review_queue(case, limit=1, offset=1)) == 1
    assert svc.review_queue(case, order="score")[0]["score"] >= svc.review_queue(case, order="score")[1]["score"]


def test_recorded_search_cap_and_missing_camera(svc, case_ref, tmp_path, monkeypatch):
    case, ref = case_ref
    clip = svc.add_original(O, case, _clip(tmp_path / "c.mp4", [RED, BLUE, RED]), source_description="s",
                            source_camera_id="CAM1", make_proxy=False)
    job = svc.create_search_job(O, case, mode="RECORDED", reference_media_id=ref, authority_ref="FIR 123/2026",
                                clip_media_ids=[clip])
    monkeypatch.setattr(face_search, "MAX_CANDIDATES_PER_JOB", 1)
    assert svc.run_recorded_search(O, job)["status"] == "CAPPED"
    assert svc.get_job(job)["candidates_found"] == 1
    nocam = svc.add_original(O, case, _clip(tmp_path / "d.mp4", [RED]), source_description="s", make_proxy=False)
    j2 = svc.create_search_job(O, case, mode="RECORDED", reference_media_id=ref, authority_ref="FIR 123/2026",
                               clip_media_ids=[nocam])
    r = svc.run_recorded_search(O, j2)
    assert r["status"] == "FAILED" and "source camera" in r["error"]


def _wait(cond, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_live_search_flags_only_matching_camera_once_then_stops(tmp_path):
    src = FakeSource({"CAM1": np.full((64, 64, 3), RED, np.uint8), "CAM2": np.full((64, 64, 3), BLUE, np.uint8)})
    s = CaseService(str(tmp_path / "cw.db"), str(tmp_path / "media"), matcher=FakeMatcher(),
                    frame_source_factory=lambda: src, live_interval_s=0.05)
    _cameras(s)
    case = s.create_case(O, "L-1", "live", "PHOTO")
    ref = s.add_original(O, case, _solid(tmp_path / "ref.png", RED), source_description="reference")
    job = s.create_search_job(O, case, mode="LIVE", reference_media_id=ref, authority_ref="FIR 9/2026",
                              camera_ids=["CAM1", "CAM2"], duration_hours=1)
    s.start_live_search(O, job)
    assert _wait(lambda: s.get_job(job)["candidates_found"] >= 1 and s.get_job(job)["frames_scanned"] >= 6)
    s.stop_search(O, job)
    done = s.get_job(job)
    assert done["status"] == "STOPPED" and done["candidates_found"] == 1 and src.closed   # 30 s cooldown per camera
    q = s.review_queue(case)
    assert [r["camera_id"] for r in q] == ["CAM1"] and q[0]["status"] == "PENDING"
    snap = s.get_media(q[0]["snapshot_media_id"])
    assert snap["role"] == "ORIGINAL" and snap["source_camera_id"] == "CAM1"


def test_live_search_expires(tmp_path):
    s = CaseService(str(tmp_path / "cw.db"), str(tmp_path / "media"), matcher=FakeMatcher(),
                    frame_source_factory=lambda: FakeSource({}), live_interval_s=0.05)
    _cameras(s)
    case = s.create_case(O, "L-2", "live", "PHOTO")
    ref = s.add_original(O, case, _solid(tmp_path / "ref.png", RED), source_description="reference")
    job = s.create_search_job(O, case, mode="LIVE", reference_media_id=ref, authority_ref="FIR 9/2026",
                              camera_ids=["CAM1"], duration_hours=0.00005)
    s.start_live_search(O, job)
    assert _wait(lambda: s.get_job(job)["status"] == "EXPIRED")
    assert s.get_job(job)["finished_at"]
    assert s.resume_after_restart() == 0


def test_live_search_needs_a_configured_source(tmp_path):
    s = CaseService(str(tmp_path / "cw.db"), str(tmp_path / "media"), matcher=FakeMatcher())
    _cameras(s)
    case = s.create_case(O, "L-3", "live", "PHOTO")
    ref = s.add_original(O, case, _solid(tmp_path / "ref.png", RED), source_description="reference")
    job = s.create_search_job(O, case, mode="LIVE", reference_media_id=ref, authority_ref="FIR 9/2026", camera_ids=["CAM1"])
    with pytest.raises(ValueError, match="frame source"):
        s.start_live_search(O, job)


def test_search_api_and_paged_queue(svc, case_ref, tmp_path, monkeypatch):
    case, ref = case_ref
    monkeypatch.setattr(routes_casework, "get_service", lambda: svc)
    app = FastAPI()
    app.include_router(routes_casework.router)
    c = TestClient(app)
    clip = svc.add_original(O, case, _clip(tmp_path / "c.mp4", [RED, BLUE, RED]), source_description="s",
                            source_camera_id="CAM1", make_proxy=False)
    bad = c.post(f"/api/casework/cases/{case}/searches", json={"mode": "LIVE", "reference_media_id": ref,
                                                              "authority_ref": "", "camera_ids": ["CAM1"]})
    assert bad.status_code == 422
    job = c.post(f"/api/casework/cases/{case}/searches", json={"mode": "RECORDED", "reference_media_id": ref,
                 "authority_ref": "FIR 123/2026", "clip_media_ids": [clip]}).json()["job_id"]
    assert c.post(f"/api/casework/searches/{job}/start").status_code == 200
    assert _wait(lambda: c.get(f"/api/casework/searches/{job}").json()["status"] == "COMPLETED")
    page = c.get(f"/api/casework/cases/{case}/review-queue", params={"page": 1, "page_size": 1, "order": "score"}).json()
    assert page["total"] == 2 and page["pages"] == 2 and len(page["items"]) == 1
    cid = page["items"][0]["candidate_id"]
    assert c.post(f"/api/casework/candidates/{cid}/decision", json={"decision": "ACCEPT", "reason": "lead"}).status_code == 200
    assert c.get(f"/api/casework/cases/{case}/review-queue", params={"status": "PENDING"}).json()["total"] == 1
    assert c.post(f"/api/casework/searches/{job}/start").status_code == 422


VOL = glob.glob(os.path.expanduser("~/Downloads/WhatsApp Unknown*/*.mp4"))


@pytest.mark.skipif(len(VOL) != 4, reason="volunteer videos not available")
def test_real_sface_on_volunteers_finds_only_the_right_clip(tmp_path):
    try:
        face_search.find_models_dir()
    except FileNotFoundError:
        pytest.skip("face models not available")
    photos = glob.glob("/tmp/claude-*/**/images/1.jpg", recursive=True)
    if not photos:
        pytest.skip("volunteer reference photo not available")
    s = CaseService(str(tmp_path / "cw.db"), str(tmp_path / "media"))
    _cameras(s)
    case = s.create_case(O, "VOL", "v", "VIDEO")
    ref = s.add_original(O, case, photos[0], source_description="volunteer A")
    clips = {os.path.basename(p): s.add_original(O, case, p, source_description="v", source_camera_id="CAM1",
                                                       make_proxy=False) for p in VOL}
    job = s.create_search_job(O, case, mode="RECORDED", reference_media_id=ref, authority_ref="FIR TEST/2026",
                              clip_media_ids=list(clips.values()), threshold=0.5)
    s.run_recorded_search(O, job)
    by_clip = {}
    for r in s.review_queue(case):
        by_clip.setdefault(r["media_id"], 0)
        by_clip[r["media_id"]] += 1
    a_front = [m for k, m in clips.items() if k.endswith("11.58.45 AM.mp4")]
    assert set(by_clip) == set(a_front)
