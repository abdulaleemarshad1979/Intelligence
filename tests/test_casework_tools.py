"""Tests for Scenario A/B tools, descriptors, the accuracy harness and the casework API."""

import glob
import os
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes_casework
from app.casework import evaluation as ev
from app.casework.media_store import probe_video
from app.casework.schema import connect
from app.casework.service import CaseService
from app.compliance.rbac import DEFAULT_OFFICER as O
from tests.test_casework import _image, _vfr_video

DOWNLOADS = glob.glob(os.path.expanduser("~/Downloads/WhatsApp Unknown*/*.mp4"))
MODELS = "/home/abdul-aleem-arshad/WORK/cctv-intelligence/models"


@pytest.fixture
def svc(tmp_path):
    return CaseService(str(tmp_path / "cw.db"), str(tmp_path / "media"))


@pytest.fixture
def photo_case(svc, tmp_path):
    case = svc.create_case(O, "T-A", "photo", "PHOTO")
    orig = svc.add_original(O, case, _image(tmp_path / "ref.png", size=(400, 300)), source_description="s")
    return case, orig


@pytest.fixture
def video_case(svc, tmp_path):
    case = svc.create_case(O, "T-B", "video", "VIDEO")
    v = svc.add_original(O, case, _vfr_video(tmp_path / "w.mp4"), source_description="s")
    return case, v


def test_crop_and_enhance_are_derived_logged_and_original_untouched(svc, photo_case):
    case, orig = photo_case
    before = svc.get_media(orig)["sha256"]
    crop = svc.crop_image(O, orig, 10, 10, 100, 100)
    enh = svc.enhance_image(O, crop, "CLAHE", clip_limit=3.0)
    up = svc.enhance_image(O, enh, "UPSCALE_BICUBIC", factor=2.0)
    assert svc.get_media(up)["width"] == 200
    chain = svc.lineage(up)
    assert [s["derivation"] and s["derivation"]["operation"] for s in chain] == [None, "CROP", "CLAHE", "UPSCALE_BICUBIC"]
    assert chain[2]["derivation"]["parameters"] == '{"clip_limit": 3.0}'
    assert svc.get_media(orig)["sha256"] == before
    svc.verify_media(orig)
    with pytest.raises(ValueError):
        svc.crop_image(O, orig, 350, 0, 100, 100)
    with pytest.raises(ValueError):
        svc.enhance_image(O, orig, "FACE_RESTORE_GAN")


def test_compare_images_records_both_sources(svc, photo_case, tmp_path):
    case, a = photo_case
    b = svc.add_original(O, case, _image(tmp_path / "b.png", 90, (300, 400)), source_description="s")
    sheet = svc.compare_images(O, a, b)
    d = svc.lineage(sheet)[-1]["derivation"]
    assert b in d["parameters"] and svc.get_media(sheet)["label"].startswith("DERIVED COPY")
    assert svc.get_media(sheet)["height"] == 640 + 28


def test_frame_step_contact_sheet_slow_motion_and_pairing(svc, video_case, tmp_path):
    case, v = video_case
    times = svc.frame_times(v)
    still = svc.extract_frame(O, v, 3)
    assert svc.get_media(still)["kind"] == "IMAGE"
    d = svc.lineage(still)[-1]["derivation"]
    assert '"frame_no": 3' in d["parameters"] and f'{times[3]}' in d["parameters"]
    with pytest.raises(ValueError):
        svc.extract_frame(O, v, 999)
    sheet = svc.contact_sheet(O, v, 0.0, times[-1], count=6, cols=3)
    assert svc.get_media(sheet)["kind"] == "IMAGE"
    normal = svc.get_media(svc.slow_motion(O, v, 0.0, 1.0, speed=1.0))["duration_s"]
    half = svc.get_media(svc.slow_motion(O, v, 0.0, 1.0, speed=0.5))["duration_s"]
    assert half == pytest.approx(2 * normal, rel=0.2)
    pair = svc.compare_frames(O, v, 3, v, 1)
    assert svc.get_media(pair)["role"] == "DERIVED"
    both = svc.compare_videos(O, v, v, speed=0.5)
    assert svc.get_media(both)["kind"] == "VIDEO" and svc.get_media(both)["width"] >= 2 * 64 - 2
    proxy = svc._conn().execute("SELECT media_id FROM cw_media WHERE role='PROXY'").fetchone()[0]
    with pytest.raises(ValueError):
        svc.slow_motion(O, proxy, 0, 1)


def test_descriptors_are_attributed_and_only_retractable(svc, photo_case):
    case, media = photo_case
    did = svc.add_descriptor(O, case, "Subject 1", "EYEWEAR", "Clear-frame glasses", media_id=media)
    svc.add_descriptor(O, case, "Subject 1", "HAIR", "Dark, curly", confidence="HIGH")
    with pytest.raises(ValueError):
        svc.add_descriptor(O, case, "Subject 1", "HEIGHT_ESTIMATE", "172 cm")
    svc.add_descriptor(O, case, "Subject 1", "HEIGHT_ESTIMATE", "170-175 cm", confidence="LOW",
                       note="against door frame, assumed 2.1 m")
    with pytest.raises(sqlite3.DatabaseError):
        svc.add_descriptor(O, case, "Subject 1", "SHOE_SIZE", "9")
    with connect(svc.db_path) as c:
        with pytest.raises(sqlite3.DatabaseError):
            c.execute("UPDATE cw_descriptors SET value='x'")
    svc.retract_descriptor(O, did)
    assert {d["category"] for d in svc.descriptors(case)} == {"HAIR", "HEIGHT_ESTIMATE"}
    assert len(svc.descriptors(case, include_retracted=True)) == 3


def test_accuracy_harness_counts_errors_and_warns_on_small_samples():
    trials = [ev.Trial(0.8, True), ev.Trial(0.7, True), ev.Trial(0.2, True, "oblique"),
              ev.Trial(0.1, False), ev.Trial(0.4, False, "oblique"), ev.Trial(0.2, False)]
    r = ev.rates_at(trials, 0.363)
    assert r["misses"] == 1 and r["false_matches"] == 1
    assert r["miss_rate"] == pytest.approx(1 / 3) and r["false_match_rate"] == pytest.approx(1 / 3)
    rep = ev.evaluate(trials)
    assert rep["lowest_threshold_with_no_false_match"] == 0.45
    assert rep["false_match_rate_95pct_upper_bound_if_zero"] == 1.0
    by = ev.evaluate_by_condition(trials, 0.363)
    assert by["oblique"]["false_matches"] == 1 and by["all"]["false_matches"] == 0
    with pytest.raises(ValueError):
        ev.evaluate([])


def test_api_round_trip(svc, tmp_path, monkeypatch):
    monkeypatch.setattr(routes_casework, "get_service", lambda: svc)
    app = FastAPI()
    app.include_router(routes_casework.router)
    c = TestClient(app)
    case = c.post("/api/casework/cases", json={"case_number": "API-1", "title": "t", "scenario": "PHOTO"}).json()["case_id"]
    img = _image(tmp_path / "up.png", size=(300, 300))
    with open(img, "rb") as f:
        up = c.post(f"/api/casework/cases/{case}/media", files={"file": ("up.png", f, "image/png")},
                    data={"source_description": "api test"})
    mid = up.json()["media_id"]
    crop = c.post(f"/api/casework/media/{mid}/crop", json={"x": 0, "y": 0, "w": 50, "h": 50})
    assert crop.status_code == 200
    assert c.post(f"/api/casework/media/{mid}/crop", json={"x": 0, "y": 0, "w": 5000, "h": 50}).status_code == 422
    assert c.post(f"/api/casework/media/{mid}/enhance", json={"operation": "GAMMA", "params": {"gamma": 1.4}}).status_code == 200
    assert c.get(f"/api/casework/media/{crop.json()['media_id']}/lineage").json()[0]["media"]["media_id"] == mid
    assert c.get("/api/casework/media/nope").status_code == 404
    bad = c.post(f"/api/casework/cases/{case}/descriptors",
                 json={"subject_label": "S1", "category": "HEIGHT_ESTIMATE", "value": "170"})
    assert bad.status_code == 422
    ok = c.post(f"/api/casework/cases/{case}/descriptors",
                json={"subject_label": "S1", "category": "HAIR", "value": "dark"})
    assert ok.status_code == 200 and len(c.get(f"/api/casework/cases/{case}/descriptors").json()) == 1
    assert c.get(f"/api/casework/media/{mid}/file").content == open(svc.get_media(mid)["storage_path"], "rb").read()


@pytest.mark.skipif(len(DOWNLOADS) != 4 or not os.path.exists(MODELS + "/face_recognition_sface_2021dec.onnx"),
                    reason="volunteer sample set / face models not available")
def test_volunteer_sample_set_end_to_end(svc, tmp_path):
    """Real volunteer clips: VFR timing, contact sheet of a walk, and the accuracy harness."""
    case = svc.create_case(O, "VOL", "volunteers", "VIDEO")
    vids = {os.path.basename(p): svc.add_original(O, case, p, source_description="volunteer", make_proxy=False)
            for p in DOWNLOADS}
    assert any(svc.get_media(m)["is_vfr"] for m in vids.values())
    walk = vids[[k for k in vids if k.endswith("11.58.39 AM.mp4")][0]]
    sheet = svc.contact_sheet(O, walk, 0.0, svc.frame_times(walk)[-1], count=8)
    assert svc.get_media(sheet)["kind"] == "IMAGE"
