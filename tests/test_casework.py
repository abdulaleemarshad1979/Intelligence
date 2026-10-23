"""Tests for the casework layer: custody, derivations, VFR timing, review decisions, camera inventory."""

import os
import sqlite3
import subprocess

import pytest

from app.casework import cameras as cam_inv
from app.casework.media_store import MediaIntegrityError, assess_image_quality
from app.casework.schema import connect
from app.casework.service import CaseService
from app.compliance.rbac import DEFAULT_OFFICER

SHEET = os.path.expanduser("~/Downloads/Copy of Matrix 544 DT. 10.09.2026（1）.xlsx")
OFFICER = DEFAULT_OFFICER


@pytest.fixture
def svc(tmp_path):
    return CaseService(str(tmp_path / "cw.db"), str(tmp_path / "media"))


def _image(path, value=128, size=(320, 320), noise=True):
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(1)
    arr = np.full((size[1], size[0]), value, dtype=np.uint8)
    if noise:
        arr = np.clip(arr + rng.integers(-40, 40, arr.shape), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return str(path)


def _vfr_video(path):
    """Frames shown at irregular times: 0, .1, .2, .5, .6, 1.5 s."""
    pts = [0, 0.1, 0.2, 0.5, 0.6, 1.5]
    fdir = path.parent / "frames"
    fdir.mkdir()
    from PIL import Image
    lines = []
    for i, t in enumerate(pts):
        Image.new("RGB", (64, 64), (i * 40, 50, 90)).save(fdir / f"f{i}.png")
        lines.append(f"file 'frames/f{i}.png'\nduration {(pts[i + 1] - t) if i + 1 < len(pts) else 0.1}")
    lst = path.parent / "list.txt"
    lst.write_text("\n".join(lines) + f"\nfile 'frames/f{len(pts) - 1}.png'\n")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-fps_mode", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


def test_original_is_preserved_and_hashed(svc, tmp_path):
    case = svc.create_case(OFFICER, "C-1", "Photo lookup", "PHOTO")
    src = _image(tmp_path / "ref.jpg")
    before = open(src, "rb").read()
    mid = svc.add_original(OFFICER, case, src, source_description="Family supplied photo",
                           last_known_time=1_700_000_000.0, last_known_location="Bus stand")
    m = svc.get_media(mid)
    assert m["role"] == "ORIGINAL" and len(m["sha256"]) == 64
    assert open(m["storage_path"], "rb").read() == before
    assert open(src, "rb").read() == before
    assert not os.access(m["storage_path"], os.W_OK)
    svc.verify_media(mid)
    os.chmod(m["storage_path"], 0o644)
    with open(m["storage_path"], "ab") as f:
        f.write(b"x")
    with pytest.raises(MediaIntegrityError):
        svc.verify_media(mid)


def test_custody_fields_cannot_be_edited_or_deleted(svc, tmp_path):
    case = svc.create_case(OFFICER, "C-2", "t", "PHOTO")
    mid = svc.add_original(OFFICER, case, _image(tmp_path / "a.png"), source_description="s")
    with connect(svc.db_path) as c:
        with pytest.raises(sqlite3.DatabaseError):
            c.execute("UPDATE cw_media SET sha256='0' WHERE media_id=?", (mid,))
        with pytest.raises(sqlite3.DatabaseError):
            c.execute("DELETE FROM cw_media WHERE media_id=?", (mid,))


def test_derived_copy_is_labelled_logged_and_traceable(svc, tmp_path):
    case = svc.create_case(OFFICER, "C-3", "t", "PHOTO")
    orig = svc.add_original(OFFICER, case, _image(tmp_path / "a.png"), source_description="s")
    crop = svc.add_derived(OFFICER, orig, _image(tmp_path / "crop.png", size=(100, 100)),
                           operation="CROP", parameters={"box": [0, 0, 100, 100]}, tool="manual")
    sharp = svc.add_derived(OFFICER, crop, _image(tmp_path / "sharp.png", 140, (100, 100)),
                            operation="UNSHARP_MASK", parameters={"radius": 1.2})
    m = svc.get_media(sharp)
    assert m["role"] == "DERIVED" and m["label"].startswith("DERIVED COPY")
    chain = svc.lineage(sharp)
    assert [s["media"]["media_id"] for s in chain] == [orig, crop, sharp]
    assert [s["derivation"] and s["derivation"]["operation"] for s in chain] == [None, "CROP", "UNSHARP_MASK"]
    svc.verify_media(orig)
    with connect(svc.db_path) as c:
        with pytest.raises(sqlite3.DatabaseError):
            c.execute("UPDATE cw_derivations SET operation='X'")


def test_quality_warnings(tmp_path):
    dark = assess_image_quality(_image(tmp_path / "d.png", value=15))
    assert "UNDEREXPOSED" in dark["warnings"]
    flat = assess_image_quality(_image(tmp_path / "f.png", value=128, noise=False))
    assert "BLURRY" in flat["warnings"]
    small = assess_image_quality(_image(tmp_path / "s.png", size=(80, 80)))
    assert "LOW_RESOLUTION" in small["warnings"] and "OCCLUSION_NOT_ASSESSED" in small["warnings"]


@pytest.mark.skipif(not subprocess.run(["which", "ffmpeg"], capture_output=True).stdout, reason="ffmpeg missing")
def test_vfr_timestamps_and_proxy(svc, tmp_path):
    case = svc.create_case(OFFICER, "C-4", "walk", "VIDEO")
    vid = _vfr_video(tmp_path / "walk.mp4")
    mid = svc.add_original(OFFICER, case, vid, source_description="Walking clip")
    m = svc.get_media(mid)
    assert m["is_vfr"] == 1
    times = svc.frame_times(mid)
    assert len(times) >= 6 and times[3] == pytest.approx(0.5, abs=0.05)
    assert svc.frame_at(mid, 0.55) == 3 and svc.frame_at(mid, 1.0) == 4
    proxies = [r for r in connect(svc.db_path).execute(
        "SELECT * FROM cw_media WHERE role='PROXY' AND parent_media_id=?", (mid,))]
    assert len(proxies) == 1 and proxies[0]["label"].startswith("VIEWING PROXY")
    svc.annotate(OFFICER, mid, "INTERVAL", start_s=0.2, end_s=1.0, label="walking")
    svc.annotate(OFFICER, mid, "PERSON", start_s=0.2, end_s=0.6, bbox={"x": .1, "y": .1, "w": .3, "h": .6})
    with pytest.raises(ValueError):
        svc.annotate(OFFICER, mid, "INTERVAL", start_s=0.2, end_s=99)
    assert len(svc.annotations(mid)) == 2


def test_review_queue_records_every_decision(svc, tmp_path):
    case = svc.create_case(OFFICER, "C-5", "t", "PHOTO")
    ref = svc.add_original(OFFICER, case, _image(tmp_path / "a.png"), source_description="s",
                           last_known_time=1_700_000_000.0, last_known_latitude=16.9984582,
                           last_known_longitude=81.7795421)
    with connect(svc.db_path) as c:
        cam_inv.import_camera_sheet(c, SHEET) if os.path.exists(SHEET) else c.execute(
            "INSERT INTO cw_cameras VALUES ('MTX-1',1,'10.0.0.1','FIXED','Spot','PS','Z',16.9984,81.7795,'','','[]','t',0)")
    cands = svc.propose_candidates(OFFICER, case, ref, radius_m=300)
    assert cands
    cid = cands[0]
    assert svc.review_queue(case, "PENDING")
    svc.record_decision(OFFICER, cid, "DEFER", "need better angle")
    svc.record_decision(OFFICER, cid, "ACCEPT", "clothing and route consistent")
    with pytest.raises(ValueError):
        svc.record_decision(OFFICER, cid, "REJECT", "")
    assert [d["decision"] for d in svc.decision_history(cid)] == ["DEFER", "ACCEPT"]
    assert next(r for r in svc.review_queue(case) if r["candidate_id"] == cid)["status"] == "ACCEPT"
    with connect(svc.db_path) as c:
        with pytest.raises(sqlite3.DatabaseError):
            c.execute("DELETE FROM cw_review_decisions")


@pytest.mark.skipif(not os.path.exists(SHEET), reason="camera sheet not available")
def test_camera_sheet_import(tmp_path):
    from app.casework.schema import init_casework_db
    path = init_casework_db(str(tmp_path / "c.db"))
    conn = connect(path)
    res = cam_inv.import_camera_sheet(conn, SHEET)
    assert res["rows"] == 544 and res["inserted"] == 544
    assert conn.execute("SELECT COUNT(DISTINCT ip_address) FROM cw_cameras").fetchone()[0] == 544
    assert conn.execute("SELECT COUNT(*) FROM cw_cameras WHERE latitude IS NULL").fetchone()[0] == 1
    again = cam_inv.import_camera_sheet(conn, SHEET)
    assert again["inserted"] == 0 and again["updated"] == 544
    near = cam_inv.cameras_near(conn, 16.9984582, 81.7795421, 100)
    assert near and near[0]["distance_m"] < 100
