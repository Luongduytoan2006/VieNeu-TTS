"""Async TTS jobs: create → poll → cancel → download."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, Response

from ..config import settings

from ..db import session_scope
from ..services.jobs import manager
from ..models import (
    ACTIVE_STATUSES, STATUS_CANCELLED, STATUS_DONE, STATUS_QUEUED, Job,
)
from ..schemas import DEFAULT_STYLE, STYLE_CHOICES, ErrorResponse, JobCreated, JobStatus, TTSCreate
from ..services.tts_service import service
from ..services import catalog_provider as catalog

router = APIRouter(prefix="/tts", tags=["tts"])

API_PREFIX = "/api/v1"


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
    """Tạo job. Trả về **job id (uuid)** ngay; tổng hợp chạy nền.

    Poll ``GET /api/v1/tts/{id}`` để xem % tiến độ, ``DELETE`` để hủy,
    ``GET /api/v1/tts/{id}/download`` để tải WAV khi xong.
    """
    if not catalog.ready():
        raise HTTPException(503, "Model chưa sẵn sàng. Xem GET /api/v1/health.")

    voice = req.voice
    if voice and not catalog.has_voice(voice):
        raise HTTPException(422, f"Voice '{voice}' không tồn tại. Xem GET {API_PREFIX}/voices.")
    style = req.style if req.style in STYLE_CHOICES else DEFAULT_STYLE

    with session_scope() as s:
        job = Job(text=req.text, voice=voice, style=style,
                  temperature=req.temperature, status=STATUS_QUEUED)
        s.add(job)
        s.flush()
        job_id = job.id

    manager.start(job_id)
    return JobCreated(id=job_id, status=STATUS_QUEUED,
                      poll_url=f"{API_PREFIX}/tts/{job_id}", download_url=_download_url(job_id))


@router.get("/{job_id}", response_model=JobStatus,
            responses={404: {"model": ErrorResponse}},
            summary="Trạng thái + % tiến độ của job")
def get_tts(job_id: str) -> JobStatus:
    with session_scope() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, f"Job '{job_id}' không tồn tại.")
        return _to_status(job)


@router.delete("/{job_id}", response_model=JobStatus,
               responses={404: {"model": ErrorResponse}},
               summary="Hủy job đang chạy (khách đổi ý)")
def cancel_tts(job_id: str) -> JobStatus:
    with session_scope() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, f"Job '{job_id}' không tồn tại.")
        was_active = job.status in ACTIVE_STATUSES

    if was_active:
        manager.cancel(job_id)
        # Best-effort: reflect cancellation immediately if the worker hasn't yet.
        with session_scope() as s:
            job = s.get(Job, job_id)
            if job and job.status in ACTIVE_STATUSES:
                job.status = STATUS_CANCELLED

    with session_scope() as s:
        return _to_status(s.get(Job, job_id))


@router.get("/{job_id}/download",
            responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
            summary="Tải file WAV kết quả")
def download_tts(job_id: str):
    with session_scope() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, f"Job '{job_id}' không tồn tại.")
        if job.status != STATUS_DONE:
            raise HTTPException(409, f"Job chưa xong (status={job.status}). Poll {API_PREFIX}/tts/{job_id}.")
        local_path = job.audio_path
        remote_url = job.audio_url

    # Local mode: file on the BE's disk.
    if local_path:
        return FileResponse(local_path, media_type="audio/wav", filename=f"vieneu_{job_id}.wav")
    # Remote mode (PA3): audio lives on the model-server storage. Proxy it so the
    # caller always uses the same BE URL (and doesn't need to reach the GPU box).
    if remote_url:
        from ..services.model_client import client
        try:
            data = client.fetch_audio(remote_url)
        except Exception as e:
            raise HTTPException(502, f"Không lấy được audio từ storage: {e}")
        return Response(content=data, media_type="audio/wav",
                        headers={"Content-Disposition": f'inline; filename="vieneu_{job_id}.wav"'})
    raise HTTPException(409, "Job xong nhưng không có audio.")
