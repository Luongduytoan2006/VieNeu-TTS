"""Jobs management routes — list/detail/update/delete persisted TTS jobs."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..schemas import (
    ErrorResponse, JobDeleteResponse, JobInfo, JobsResponse, JobUpdate,
)
from ..services import job_admin as job_admin_svc

router = APIRouter(prefix=settings.API_PREFIX, tags=["jobs"])


@router.get("/jobs", response_model=JobsResponse,
            summary="GET /api/v1/jobs — Danh sách toàn bộ job trong DB")
def list_jobs() -> JobsResponse:
    return job_admin_svc.list_all()


@router.get("/jobs/users/{user_ref}", response_model=JobsResponse,
            summary="GET /api/v1/jobs/users/{user_ref} — Danh sách job theo user_ref")
def list_jobs_by_user(user_ref: str) -> JobsResponse:
    return job_admin_svc.list_user(user_ref)


@router.get("/jobs/{job_id}", response_model=JobInfo,
            responses={404: {"model": ErrorResponse}},
            summary="GET /api/v1/jobs/{job_id} — Chi tiết 1 job trong DB")
def get_job(job_id: str) -> JobInfo:
    try:
        return job_admin_svc.get(job_id)
    except job_admin_svc.JobAdminError as e:
        raise HTTPException(e.status, e.detail)


@router.patch("/jobs/{job_id}", response_model=JobInfo,
              responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
              summary="PATCH /api/v1/jobs/{job_id} — Đổi tên hiển thị file audio")
def update_job(job_id: str, req: JobUpdate) -> JobInfo:
    try:
        return job_admin_svc.update_name(job_id, req.name_audio)
    except job_admin_svc.JobAdminError as e:
        raise HTTPException(e.status, e.detail)


@router.delete("/jobs/{job_id}", response_model=JobDeleteResponse,
               responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
               summary="DELETE /api/v1/jobs/{job_id} — Xóa job DB và audio storage")
def delete_job(job_id: str) -> JobDeleteResponse:
    try:
        return job_admin_svc.delete(job_id)
    except job_admin_svc.JobAdminError as e:
        raise HTTPException(e.status, e.detail)
