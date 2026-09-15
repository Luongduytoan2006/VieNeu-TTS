"""TTS routes — create, poll, cancel, and download async synthesis jobs."""
from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import Response

from ..config import settings
from ..schemas import ErrorResponse, JobCreated, JobStatus, TTSCreate
from ..services import create_audio
from ..services.jobs import ACTIVE_STATUSES, DONE, Job, manager

router = APIRouter(prefix=settings.API_PREFIX, tags=["tts"])

_UID = Header(default=None, alias="X-User-Id", description="ID người dùng (module ngoài truyền vào).")


def _user(x_user_id: Optional[str]) -> str:
    uid = (x_user_id or "").strip()
    return uid or "default"


def _download_url(job: Job) -> Optional[str]:
    if getattr(job, "delivery_kind", "vieneu") == "openvoice":
        return None
    return f"{settings.API_PREFIX}/tts/{job.id}/download"


def _to_status(job: Job) -> JobStatus:
    inst_id = dph = est_cost = None
    if job.mode == "gpu":
        from ..services import gpu_vastai
        gs = gpu_vastai.get_status(job)
        inst_id = gs.get("instance_id") or getattr(job, "instance_id", None)
        dph = gs.get("dph") or None
        est_cost = gs.get("est_cost") or None
    return JobStatus(
        id=job.id, status=job.status, mode=job.mode, progress=job.progress,
        done_chunks=job.done_chunks, total_chunks=job.total_chunks,
        voice=job.voice, style=job.style, name_audio=job.name_audio,
        audio_key=job.audio_key, audio_size_bytes=job.audio_size_bytes,
        duration_sec=job.duration_sec, elapsed_sec=job.elapsed_sec,
        sample_rate=job.sample_rate, error=job.error,
        delivery_kind=getattr(job, "delivery_kind", "vieneu"),
        external_generation_id=getattr(job, "external_generation_id", None),
        delivery_unconfirmed=getattr(job, "delivery_unconfirmed", False),
        instance_id=inst_id, dph=dph, est_cost=est_cost,
        download_url=_download_url(job) if job.status == DONE else None,
        created_at=job.created_at, updated_at=job.updated_at,
    )


@router.post("/tts", response_model=JobCreated, status_code=202,
             responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
             summary="POST /api/v1/tts — Tạo job tổng hợp audio (bất đồng bộ, cpu|gpu)")
def create_tts(req: TTSCreate, x_user_id: Optional[str] = _UID) -> JobCreated:
    try:
        job = create_audio.create(req.text, req.voice, req.style, req.temperature,
                                  req.max_chars, mode=req.mode, user_ref=_user(x_user_id),
                                  name_audio=req.name_audio, delivery=req.delivery)
    except create_audio.CreateError as e:
        raise HTTPException(e.status, e.detail)
    return JobCreated(id=job.id, status=job.status, mode=job.mode,
                      name_audio=job.name_audio, audio_key=job.audio_key,
                      audio_size_bytes=job.audio_size_bytes,
                      delivery_kind=getattr(job, "delivery_kind", "vieneu"),
                      external_generation_id=getattr(job, "external_generation_id", None),
                      created_at=job.created_at, updated_at=job.updated_at,
                      poll_url=f"{settings.API_PREFIX}/tts/{job.id}",
                      download_url=_download_url(job))


def _owned_job(job_id: str, x_user_id: Optional[str]) -> Job:
    job = manager.get(job_id)
    if job is None or getattr(job, "user_ref", "default") != _user(x_user_id):
        raise HTTPException(404, f"Job '{job_id}' không tồn tại.")
    return job


@router.get("/tts/by-generation/{generation_id}", response_model=JobStatus,
            responses={404: {"model": ErrorResponse}},
            summary="Resolve an OpenVoice generation to its VieNeu job")
def get_tts_by_generation(generation_id: UUID,
                          x_user_id: Optional[str] = _UID) -> JobStatus:
    job = manager.get_by_generation(_user(x_user_id), str(generation_id))
    if job is None:
        raise HTTPException(404, "Generation không tồn tại.")
    return _to_status(job)


@router.get("/tts/{job_id}", response_model=JobStatus,
            responses={404: {"model": ErrorResponse}},
            summary="GET /api/v1/tts/{job_id} — Trạng thái + % tiến độ của job")
def get_tts(job_id: str, x_user_id: Optional[str] = _UID) -> JobStatus:
    return _to_status(_owned_job(job_id, x_user_id))


@router.delete("/tts/{job_id}", response_model=JobStatus,
               responses={404: {"model": ErrorResponse}},
               summary="DELETE /api/v1/tts/{job_id} — Hủy job đang chạy")
def cancel_tts(job_id: str, x_user_id: Optional[str] = _UID) -> JobStatus:
    job = _owned_job(job_id, x_user_id)
    if job.status in ACTIVE_STATUSES:
        manager.cancel(job_id)
    return _to_status(manager.get(job_id))


@router.get("/tts/{job_id}/download",
            responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
            summary="GET /api/v1/tts/{job_id}/download — Tải file WAV kết quả")
def download_tts(job_id: str, x_user_id: Optional[str] = _UID):
    job = _owned_job(job_id, x_user_id)
    if getattr(job, "delivery_kind", "vieneu") == "openvoice":
        raise HTTPException(410, "Audio do OpenVoice quản lý.")
    if job.status != DONE:
        raise HTTPException(409, f"Job chưa xong (status={job.status}). Poll {settings.API_PREFIX}/tts/{job_id}.")
    if not job.audio_url:
        raise HTTPException(409, "Job xong nhưng không có audio.")

    from ..storage import get_storage
    try:
        st = get_storage()
        data = st.open(job.audio_key) if job.audio_key and st.exists(job.audio_key) \
            else _fetch_url(job.audio_url)
    except Exception as e:
        raise HTTPException(502, f"Không lấy được audio từ storage: {e}")
    return Response(content=data, media_type="audio/wav",
                    headers={"Content-Disposition": f'inline; filename="vieneu_{job_id}.wav"'})


def _fetch_url(url: str) -> bytes:
    import urllib.request
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 (URL nội bộ/presigned)
        return r.read()
