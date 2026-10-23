"""REST API for the manual face/gait review workflows (cases, media, tools, descriptors, review queue)."""

import os
import shutil
import tempfile
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.casework.cameras import cameras_near
from app.casework.service import CaseService
from app.compliance.rbac import DEFAULT_OFFICER as OFFICER

router = APIRouter(prefix="/api/casework", tags=["Casework (manual review)"])


@lru_cache(maxsize=1)
def get_service() -> CaseService:
    return CaseService()


def _call(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except KeyError as e:
        raise HTTPException(404, f"Not found: {e.args[0]}")
    except ValueError as e:
        raise HTTPException(422, str(e))


class CaseIn(BaseModel):
    case_number: str
    title: str
    scenario: str = Field(pattern="^(PHOTO|VIDEO)$")
    notes: str = ""


class CropIn(BaseModel):
    x: int
    y: int
    w: int
    h: int


class EnhanceIn(BaseModel):
    operation: str
    params: Dict[str, float] = {}


class FrameIn(BaseModel):
    frame_no: int


class SheetIn(BaseModel):
    start_s: float
    end_s: float
    count: int = 12
    cols: int = 4


class SlowIn(BaseModel):
    start_s: float
    end_s: float
    speed: float = 0.25


class CompareImagesIn(BaseModel):
    media_a: str
    media_b: str


class CompareFramesIn(BaseModel):
    video_a: str
    frame_a: int
    video_b: str
    frame_b: int


class CompareVideosIn(BaseModel):
    video_a: str
    video_b: str
    speed: float = 1.0


class AnnotationIn(BaseModel):
    kind: str = Field(pattern="^(INTERVAL|PERSON|REGION|NOTE)$")
    start_s: Optional[float] = None
    end_s: Optional[float] = None
    bbox: Optional[Dict[str, float]] = None
    label: str = ""
    note: str = ""


class DescriptorIn(BaseModel):
    subject_label: str
    category: str
    value: str
    basis: str = "OBSERVED"
    confidence: str = "MEDIUM"
    media_id: Optional[str] = None
    at_time_s: Optional[float] = None
    note: str = ""


class ProposeIn(BaseModel):
    media_id: str
    radius_m: float = 500.0
    window_before_s: float = 600.0
    window_after_s: float = 1800.0
    camera_types: Optional[List[str]] = None


class DecisionIn(BaseModel):
    decision: str = Field(pattern="^(ACCEPT|REJECT|DEFER)$")
    reason: str = ""


@router.post("/cases")
def create_case(body: CaseIn):
    cid = _call(get_service().create_case, OFFICER, body.case_number, body.title, body.scenario, body.notes)
    return {"case_id": cid}


@router.get("/cases/{case_id}")
def get_case(case_id: str):
    return _call(get_service().get_case, case_id)


@router.post("/cases/{case_id}/media")
def upload_media(case_id: str, file: UploadFile = File(...), source_description: str = Form(...),
                 last_known_time: Optional[float] = Form(None), last_known_location: Optional[str] = Form(None),
                 last_known_latitude: Optional[float] = Form(None), last_known_longitude: Optional[float] = Form(None),
                 source_camera_id: Optional[str] = Form(None)):
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, os.path.basename(file.filename or "upload"))
    try:
        with open(path, "wb") as out:
            shutil.copyfileobj(file.file, out)
        mid = _call(get_service().add_original, OFFICER, case_id, path, source_description=source_description,
                    last_known_time=last_known_time, last_known_location=last_known_location,
                    last_known_latitude=last_known_latitude, last_known_longitude=last_known_longitude,
                    source_camera_id=source_camera_id)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return {"media_id": mid, "media": get_service().get_media(mid)}


@router.get("/media/{media_id}")
def get_media(media_id: str):
    return _call(get_service().get_media, media_id)


@router.get("/media/{media_id}/file")
def media_file(media_id: str):
    m = _call(get_service().get_media, media_id)
    return FileResponse(m["storage_path"], media_type=m["mime_type"])


@router.get("/media/{media_id}/lineage")
def lineage(media_id: str):
    return _call(get_service().lineage, media_id)


@router.get("/media/{media_id}/frame-times")
def frame_times(media_id: str):
    return {"pts_seconds": _call(get_service().frame_times, media_id)}


@router.post("/media/{media_id}/quality")
def quality(media_id: str):
    return _call(get_service().assess_image, OFFICER, media_id)


@router.post("/media/{media_id}/crop")
def crop(media_id: str, b: CropIn):
    return {"media_id": _call(get_service().crop_image, OFFICER, media_id, b.x, b.y, b.w, b.h)}


@router.post("/media/{media_id}/enhance")
def enhance(media_id: str, b: EnhanceIn):
    return {"media_id": _call(get_service().enhance_image, OFFICER, media_id, b.operation, **b.params)}


@router.post("/media/{media_id}/frame")
def frame(media_id: str, b: FrameIn):
    return {"media_id": _call(get_service().extract_frame, OFFICER, media_id, b.frame_no)}


@router.post("/media/{media_id}/contact-sheet")
def contact_sheet(media_id: str, b: SheetIn):
    return {"media_id": _call(get_service().contact_sheet, OFFICER, media_id, b.start_s, b.end_s,
                              count=b.count, cols=b.cols)}


@router.post("/media/{media_id}/slow-motion")
def slow_motion(media_id: str, b: SlowIn):
    return {"media_id": _call(get_service().slow_motion, OFFICER, media_id, b.start_s, b.end_s, b.speed)}


@router.post("/compare/images")
def compare_images(b: CompareImagesIn):
    return {"media_id": _call(get_service().compare_images, OFFICER, b.media_a, b.media_b)}


@router.post("/compare/frames")
def compare_frames(b: CompareFramesIn):
    return {"media_id": _call(get_service().compare_frames, OFFICER, b.video_a, b.frame_a, b.video_b, b.frame_b)}


@router.post("/compare/videos")
def compare_videos(b: CompareVideosIn):
    return {"media_id": _call(get_service().compare_videos, OFFICER, b.video_a, b.video_b, b.speed)}


@router.post("/media/{media_id}/annotations")
def annotate(media_id: str, b: AnnotationIn):
    return {"annotation_id": _call(get_service().annotate, OFFICER, media_id, b.kind, start_s=b.start_s,
                                   end_s=b.end_s, bbox=b.bbox, label=b.label, note=b.note)}


@router.get("/media/{media_id}/annotations")
def annotations(media_id: str):
    return _call(get_service().annotations, media_id)


@router.post("/cases/{case_id}/descriptors")
def add_descriptor(case_id: str, b: DescriptorIn):
    try:
        did = get_service().add_descriptor(OFFICER, case_id, b.subject_label, b.category, b.value,
                                           basis=b.basis, confidence=b.confidence, media_id=b.media_id,
                                           at_time_s=b.at_time_s, note=b.note)
    except KeyError as e:
        raise HTTPException(404, f"Not found: {e.args[0]}")
    except Exception as e:  # ValueError or DB CHECK violations
        raise HTTPException(422, str(e))
    return {"descriptor_id": did}


@router.get("/cases/{case_id}/descriptors")
def list_descriptors(case_id: str, subject_label: Optional[str] = None, include_retracted: bool = False):
    return get_service().descriptors(case_id, subject_label, include_retracted)


@router.post("/descriptors/{descriptor_id}/retract")
def retract(descriptor_id: str):
    get_service().retract_descriptor(OFFICER, descriptor_id)
    return {"status": "RETRACTED"}


@router.get("/cameras/near")
def cameras_nearby(lat: float, lon: float, radius_m: float = 500.0, camera_type: Optional[str] = None):
    svc = get_service()
    with svc._conn() as c:
        return cameras_near(c, lat, lon, radius_m, [camera_type] if camera_type else None)


@router.post("/cases/{case_id}/candidates/propose")
def propose(case_id: str, b: ProposeIn):
    ids = _call(get_service().propose_candidates, OFFICER, case_id, b.media_id, radius_m=b.radius_m,
                window_before_s=b.window_before_s, window_after_s=b.window_after_s, camera_types=b.camera_types)
    return {"candidate_ids": ids}


@router.get("/cases/{case_id}/review-queue")
def review_queue(case_id: str, status: Optional[str] = None):
    return get_service().review_queue(case_id, status)


@router.post("/candidates/{candidate_id}/decision")
def decide(candidate_id: str, b: DecisionIn):
    return {"decision_id": _call(get_service().record_decision, OFFICER, candidate_id, b.decision, b.reason)}


@router.get("/candidates/{candidate_id}/decisions")
def decisions(candidate_id: str):
    return get_service().decision_history(candidate_id)
