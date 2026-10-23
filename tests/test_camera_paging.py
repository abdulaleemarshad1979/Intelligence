"""Server-side paging for the camera grid and the camera inventory."""

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes_casework
from app.casework import cameras as cam_inv
from app.casework.schema import connect
from app.casework.service import CaseService

SHEET = os.path.expanduser("~/Downloads/Copy of Matrix 544 DT. 10.09.2026（1）.xlsx")


@pytest.fixture(scope="module")
def grid():
    import app.main as m
    return TestClient(m.app)


def test_grid_page_is_capped_clamped_and_filtered(grid):
    d = grid.get("/cameras/page", params={"page": 2, "page_size": 24}).json()
    assert d["page"] == 2 and len(d["items"]) == 24 and d["items"][0]["id"] == "CAM-025" and d["total"] == d["fleet_total"]
    d = grid.get("/cameras/page", params={"page_size": 100000}).json()
    assert d["page_size"] == 48 and len(d["items"]) == 48
    d = grid.get("/cameras/page", params={"page": 999, "page_size": 24}).json()
    assert d["fleet_total"] == 544 and d["page"] == d["pages"] == 23 and len(d["items"]) == 544 - 22 * 24
    ip = grid.get("/cameras/page", params={"q": "10.243.0.4"}).json()      # search matches the IP
    assert ip["total"] >= 1 and ip["items"][0]["name"] == "Thadithota-2"
    assert 0 < grid.get("/cameras/page", params={"q": "thadithota"}).json()["total"] < 20


def test_grid_keeps_full_and_index_endpoints(grid):
    full, idx = grid.get("/cameras").json(), grid.get("/cameras/index").json()
    assert len(full) == len(idx) == 544 and set(idx[0]) == {"id", "name", "location", "sector", "subdivision", "ip"}


@pytest.mark.skipif(not os.path.exists(SHEET), reason="camera sheet not available")
def test_inventory_endpoint_pages_the_544_sheet(tmp_path, monkeypatch):
    svc = CaseService(str(tmp_path / "cw.db"), str(tmp_path / "media"))
    with connect(svc.db_path) as c:
        cam_inv.import_camera_sheet(c, SHEET)
    monkeypatch.setattr(routes_casework, "get_service", lambda: svc)
    app = FastAPI()
    app.include_router(routes_casework.router)
    c = TestClient(app)
    d = c.get("/api/casework/cameras", params={"page": 2, "page_size": 100}).json()
    assert d["total"] == 544 and d["pages"] == 6 and len(d["items"]) == 100
    assert c.get("/api/casework/cameras", params={"page": 6, "page_size": 100}).json()["items"].__len__() == 44
    assert c.get("/api/casework/cameras", params={"camera_type": "FRS"}).json()["total"] == 25
    assert c.get("/api/casework/cameras", params={"q": "Thadithota-1"}).json()["total"] >= 3


def test_subject_name_is_optional():
    from app.main import SuspectIntakePayload, TargetFaceEnrollPayload
    a = SuspectIntakePayload(fir_no="")
    b = TargetFaceEnrollPayload(name="  ", image_path="x.jpg")
    assert a.name.startswith("Unnamed subject ") and b.name.startswith("Unnamed subject ") and a.name != b.name
    assert SuspectIntakePayload(name=" Raju ", fir_no="1").name == "Raju"
