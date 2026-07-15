"""Async TTS jobs: create → poll → cancel → download."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ..config import settings
from ..models import ACTIVE_STATUSES, STATUS_DONE, Job
from ..schemas import (
    DEFAULT_STYLE, STYLE_CHOICES, ErrorResponse, JobCreated, JobStatus, TTSCreate,
)
from ..services import catalog_service
from ..services.job_service import fetch_audio, get_job, manager

router = APIRouter(prefix="/tts", tags=["tts"])

API_PREFIX = settings.API_PREFIX


def _download_url(job_id: str) -> str:
    return f"{API_PREFIX}/tts/{job_id}/download"


def _to_status(job: Job) -> JobStatus:
    return JobStatus(
        id=job.id, status=job.status, progress=job.progress,
        done_chunks=job.done_chunks, total_chunks=job.total_chunks,
        voice=job.voice, style=job.style, duration_sec=job.duration_sec,
        elapsed_sec=job.elapsed_sec, sample_rate=job.sample_rate, error=job.error,
        download_url=_download_url(job.id) if job.status == STATUS_DONE else None,
        created_at=job.created_at, updated_at=job.updated_at,
    )


@router.post("", response_model=JobCreated, status_code=202,
             responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
             summary="Tạo job tổng hợp audio (bất đồng bộ)")
def create_tts(req: TTSCreate) -> JobCreated:
    """Tạo job. Trả về **job id (uuid)** ngay; tổng hợp chạy nền trên model-server.

    Poll ``GET /api/v1/tts/{id}`` để xem % tiến độ, ``DELETE`` để hủy,
    ``GET /api/v1/tts/{id}/download`` để tải WAV khi xong.
    """
    if not catalog_service.ready():
        raise HTTPException(503, "Model chưa sẵn sàng. Xem GET /api/v1/health.")

    voice = req.voice
    if voice and not catalog_service.has_voice(voice):
        raise HTTPException(422, f"Voice '{voice}' không tồn tại. Xem GET {API_PREFIX}/voices.")
    style = req.style if req.style in STYLE_CHOICES else DEFAULT_STYLE

    job_id = manager.create(req.text, voice, style, req.temperature)
    return JobCreated(id=job_id, status="queued",
                      poll_url=f"{API_PREFIX}/tts/{job_id}", download_url=_download_url(job_id))


@router.get("/{job_id}", response_model=JobStatus,
            responses={404: {"model": ErrorResponse}},
            summary="Trạng thái + % tiến độ của job")
def get_tts(job_id: str) -> JobStatus:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job '{job_id}' không tồn tại.")
    return _to_status(job)


@router.delete("/{job_id}", response_model=JobStatus,
               responses={404: {"model": ErrorResponse}},
               summary="Hủy job đang chạy (khách đổi ý)")
def cancel_tts(job_id: str) -> JobStatus:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job '{job_id}' không tồn tại.")
    if job.status in ACTIVE_STATUSES:
        manager.cancel(job_id)
    job = get_job(job_id)
    return _to_status(job)


@router.get("/{job_id}/download",
            responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
            summary="Tải file WAV kết quả")
def download_tts(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job '{job_id}' không tồn tại.")
    if job.status != STATUS_DONE:
        raise HTTPException(409, f"Job chưa xong (status={job.status}). Poll {API_PREFIX}/tts/{job_id}.")
    if not job.audio_url:
        raise HTTPException(409, "Job xong nhưng không có audio.")

    # Audio lives on the model-server storage (R2 presigned / file route). Proxy it
    # so the client always uses this BE URL and never needs to reach the GPU box.
    try:
        data = fetch_audio(job.audio_url)
    except Exception as e:
        raise HTTPException(502, f"Không lấy được audio từ storage: {e}")
    return Response(content=data, media_type="audio/wav",
                    headers={"Content-Disposition": f'inline; filename="vieneu_{job_id}.wav"'})
