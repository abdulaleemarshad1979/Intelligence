"""FastAPI Router for Multi-Camera Target Person Search."""

import os
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from app.target_search.coordinator import get_target_search_coordinator
from app.compliance.rbac import OfficerIdentity, DEFAULT_OFFICER, Permission
from app.compliance.audit import AuditLogger

router = APIRouter(prefix="/api/target-search", tags=["Target Person Search"])
audit_logger = AuditLogger()


def get_current_officer() -> OfficerIdentity:
    """Resolve active officer context for audit and access control."""
    return DEFAULT_OFFICER


class StopSearchResponse(BaseModel):
    status: str
    session_id: str
    message: str


@router.post("/start")
async def start_target_search(
    image: UploadFile = File(..., description="Reference image of the target person"),
    name: Optional[str] = Form("Subject of Interest"),
    mode: Optional[str] = Form("auto"),
    cameras: Optional[str] = Form("*"),
    officer: OfficerIdentity = Depends(get_current_officer)
):
    """Initiate a multi-camera target person search session with reference representation extraction."""
    coordinator = get_target_search_coordinator()

    try:
        content = await image.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty reference image uploaded.")

        result = coordinator.start_search(
            image_input=content,
            name=name,
            mode=mode or "auto",
            cameras=cameras or "*"
        )

        # Audit log creation of target search session
        audit_logger.log_action(
            action_type="START_TARGET_SEARCH",
            resource_id=result["session_id"],
            details={
                "target_name": name,
                "requested_mode": mode,
                "cameras": result["selected_cameras"]
            },
            officer=officer
        )

        return JSONResponse(status_code=200, content=result)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to start target search: {str(ex)}")


@router.get("/{session_id}")
async def get_target_search_status(
    session_id: str,
    officer: OfficerIdentity = Depends(get_current_officer)
):
    """Retrieve runtime status, active camera counts, and telemetry for a target search session."""
    coordinator = get_target_search_coordinator()
    status = coordinator.get_status(session_id)
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])
    return JSONResponse(status_code=200, content=status)


@router.get("/{session_id}/events")
async def get_target_search_events(
    session_id: str,
    officer: OfficerIdentity = Depends(get_current_officer)
):
    """Retrieve all confirmed candidate events emitted by temporal confirmation for this session."""
    coordinator = get_target_search_coordinator()
    session = coordinator.get_session(session_id)
    if not session:
        # Check database fallback
        events = coordinator.repo.get_target_search_events_for_session(session_id)
        return JSONResponse(status_code=200, content=events)

    events = coordinator.get_events(session_id)
    return JSONResponse(status_code=200, content=events)


@router.get("/{session_id}/cameras")
async def get_target_search_cameras(
    session_id: str,
    officer: OfficerIdentity = Depends(get_current_officer)
):
    """Retrieve real-time camera connection and active track metrics for this search session."""
    coordinator = get_target_search_coordinator()
    session = coordinator.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Search session '{session_id}' not found.")

    cam_status = coordinator.get_cameras_status(session_id)
    return JSONResponse(status_code=200, content=cam_status)


@router.get("/{session_id}/evidence")
async def get_target_search_evidence(
    session_id: str,
    officer: OfficerIdentity = Depends(get_current_officer)
):
    """Retrieve forensic evidence artifacts and SHA-256 verification hashes for this session."""
    coordinator = get_target_search_coordinator()
    evidence = coordinator.get_evidence(session_id)
    return JSONResponse(status_code=200, content=evidence)


@router.post("/{session_id}/stop")
async def stop_target_search(
    session_id: str,
    officer: OfficerIdentity = Depends(get_current_officer)
):
    """Gracefully terminate a running target person search session without restarting video feeds."""
    coordinator = get_target_search_coordinator()
    success = coordinator.stop_search(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Target search session '{session_id}' not found.")

    audit_logger.log_action(
        action_type="STOP_TARGET_SEARCH",
        resource_id=session_id,
        details={"status": "Stopped"},
        officer=officer
    )

    return JSONResponse(status_code=200, content={
        "status": "SUCCESS",
        "session_id": session_id,
        "message": f"Target search session '{session_id}' stopped."
    })


@router.get("/{session_id}/evidence/{event_id}/{filename}")
async def get_evidence_file(
    session_id: str,
    event_id: str,
    filename: str,
    officer: OfficerIdentity = Depends(get_current_officer)
):
    """Serve full scene, person crop, or face crop evidence artifacts."""
    coordinator = get_target_search_coordinator()
    file_path = os.path.join(coordinator.storage_dir, session_id, event_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"Evidence file '{filename}' not found.")
    return FileResponse(file_path)
