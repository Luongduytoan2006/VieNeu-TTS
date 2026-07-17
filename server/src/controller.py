"""controller — GỘP TOÀN BỘ route /api/v1 vào 1 file (theo thiết kế).

Controller mỏng: validate nhẹ + map lỗi → HTTP, đẩy hết logic xuống ``services/``.
Đây là nơi xuất ra Swagger đầy đủ cho người dùng gọi.

Nhóm API:
  system  : GET  /health
  catalog : GET  /styles, GET /modes
  voices  : GET  /voices, GET /voices/{id}, POST /voices, DELETE /voices/{id}
  tts     : POST /tts, GET /tts/{id}, DELETE /tts/{id}, GET /tts/{id}/download
Ngoài prefix: GET /files/{key} (phục vụ audio khi storage=local).
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response

from .config import settings
from .schemas import (
    DEFAULT_STYLE, ErrorResponse, HealthResponse, JobCreated, JobStatus,
    ModesResponse, StylesResponse, TTSCreate, VoiceInfo, VoicesResponse,
)
from .services import catalog as catalog_svc
from .services import createVoice, voices as voice_svc
from .services.jobs import ACTIVE_STATUSES, DONE, Job, manager

API_PREFIX = settings.API_PREFIX
router = APIRouter(prefix=API_PREFIX)


# ── helpers ───────────────────────────────────────────────────────────────────
def _user(x_user_id: Optional[str]) -> str:
    """Định danh người dùng từ header ``X-User-Id``. Backend này là module được hệ
    thống ngoài gọi → caller truyền user id vào header. Thiếu → 'default' (dùng
    chung, hợp dev/thử nghiệm). Sau này thay bằng auth thật chỉ cần đổi hàm này."""
    uid = (x_user_id or "").strip()
    return uid or "default"


def _download_url(job_id: str) -> str:
    return f"{API_PREFIX}/tts/{job_id}/download"


def _to_status(job: Job) -> JobStatus:
    return JobStatus(
        id=job.id, status=job.status, mode=job.mode, progress=job.progress,
        done_chunks=job.done_chunks, total_chunks=job.total_chunks,
        voice=job.voice, style=job.style, duration_sec=job.duration_sec,
        elapsed_sec=job.elapsed_sec, sample_rate=job.sample_rate, error=job.error,
        download_url=_download_url(job.id) if job.status == DONE else None,
        created_at=job.created_at, updated_at=job.updated_at,
    )


# ── system ────────────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse, tags=["system"],
            summary="GET /api/v1/health — Sức khỏe + kiến trúc")
def health() -> HealthResponse:
    return catalog_svc.health()


# ── catalog ───────────────────────────────────────────────────────────────────
@router.get("/styles", response_model=StylesResponse, tags=["catalog"],
            summary="GET /api/v1/styles — Phong cách đọc")
def styles() -> StylesResponse:
    return catalog_svc.list_styles()


@router.get("/modes", response_model=ModesResponse, tags=["catalog"],
            summary="GET /api/v1/modes — Chế độ xử lý (cpu | gpu)")
def modes() -> ModesResponse:
    return catalog_svc.list_modes()


# ── voices ────────────────────────────────────────────────────────────────────
# Mọi endpoint giọng/job đọc header ``X-User-Id`` → mỗi user 1 kho riêng. Thiếu
# header = 'default'. Preset (RAM) dùng chung; custom lọc theo user.
_UID = Header(default=None, alias="X-User-Id", description="ID người dùng (module ngoài truyền vào).")


@router.get("/voices", response_model=VoicesResponse, tags=["voices"],
            summary="GET /api/v1/voices — Danh sách giọng (preset + custom của bạn)")
def list_voices(x_user_id: Optional[str] = _UID) -> VoicesResponse:
    if not catalog_svc.ready():
        raise HTTPException(503, "Model chưa sẵn sàng.")
    return voice_svc.list_voices(user_ref=_user(x_user_id))


@router.get("/voices/{voice_id}", response_model=VoiceInfo, tags=["voices"],
            responses={404: {"model": ErrorResponse}},
            summary="GET /api/v1/voices/{voice_id} — Chi tiết 1 giọng")
def get_voice(voice_id: str, x_user_id: Optional[str] = _UID) -> VoiceInfo:
    if not catalog_svc.ready():
        raise HTTPException(503, "Model chưa sẵn sàng.")
    v = voice_svc.get_voice(voice_id, user_ref=_user(x_user_id))
    if v is None:
        raise HTTPException(404, f"Voice '{voice_id}' không tồn tại.")
    return v


@router.post("/voices", response_model=VoiceInfo, status_code=201, tags=["voices"],
             responses={409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
             summary="POST /api/v1/voices — Nạp giọng mới từ audio (voice cloning)")
async def create_voice(
    name: str = Form(..., description="Tên/ID giọng mới."),
    audio: UploadFile = File(..., description="File audio mẫu 3–8s (wav/mp3/...)."),
    description: str = Form(""),
    gender: str = Form(""),
    style: str = Form(DEFAULT_STYLE),
    denoise: bool = Form(True),
    x_user_id: Optional[str] = _UID,
) -> VoiceInfo:
    data = await audio.read()
    try:
        return voice_svc.enroll(name, data, audio.filename or "ref.wav",
                                user_ref=_user(x_user_id), description=description,
                                gender=gender, style=style, denoise=denoise)
    except voice_svc.VoiceError as e:
        raise HTTPException(e.status, e.detail)


@router.delete("/voices/{voice_id}", status_code=204, tags=["voices"],
               responses={404: {"model": ErrorResponse}},
               summary="DELETE /api/v1/voices/{voice_id} — Xóa giọng custom")
def delete_voice(voice_id: str, x_user_id: Optional[str] = _UID):
    try:
        voice_svc.delete(voice_id, user_ref=_user(x_user_id))
    except voice_svc.VoiceError as e:
        raise HTTPException(e.status, e.detail)
    return None


# ── tts (async job) ───────────────────────────────────────────────────────────
@router.post("/tts", response_model=JobCreated, status_code=202, tags=["tts"],
             responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
             summary="POST /api/v1/tts — Tạo job tổng hợp audio (bất đồng bộ, cpu|gpu)")
def create_tts(req: TTSCreate, x_user_id: Optional[str] = _UID) -> JobCreated:
    """Tạo job → trả **job id** ngay. Chọn mode cpu/gpu/auto; GPU cần context dài
    (BE chặn thêm 1 lớp). Poll ``GET /tts/{id}`` xem %, ``DELETE`` để hủy,
    ``GET /tts/{id}/download`` tải WAV khi xong."""
    try:
        job = createVoice.create(req.text, req.voice, req.style, req.temperature,
                                 req.max_chars, mode=req.mode, user_ref=_user(x_user_id))
    except createVoice.CreateError as e:
        raise HTTPException(e.status, e.detail)
    return JobCreated(id=job.id, status=job.status, mode=job.mode,
                      poll_url=f"{API_PREFIX}/tts/{job.id}",
                      download_url=_download_url(job.id))


def _owned_job(job_id: str, x_user_id: Optional[str]) -> Job:
    """Lấy job + kiểm quyền: user chỉ thấy job của chính mình (404 nếu khác chủ)."""
    job = manager.get(job_id)
    if job is None or getattr(job, "user_ref", "default") != _user(x_user_id):
        raise HTTPException(404, f"Job '{job_id}' không tồn tại.")
    return job


@router.get("/tts/{job_id}", response_model=JobStatus, tags=["tts"],
            responses={404: {"model": ErrorResponse}},
            summary="GET /api/v1/tts/{job_id} — Trạng thái + % tiến độ của job")
def get_tts(job_id: str, x_user_id: Optional[str] = _UID) -> JobStatus:
    return _to_status(_owned_job(job_id, x_user_id))


@router.delete("/tts/{job_id}", response_model=JobStatus, tags=["tts"],
               responses={404: {"model": ErrorResponse}},
               summary="DELETE /api/v1/tts/{job_id} — Hủy job đang chạy")
def cancel_tts(job_id: str, x_user_id: Optional[str] = _UID) -> JobStatus:
    job = _owned_job(job_id, x_user_id)
    if job.status in ACTIVE_STATUSES:
        manager.cancel(job_id)
    return _to_status(manager.get(job_id))


@router.get("/tts/{job_id}/download", tags=["tts"],
            responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
            summary="GET /api/v1/tts/{job_id}/download — Tải file WAV kết quả")
def download_tts(job_id: str, x_user_id: Optional[str] = _UID):
    job = _owned_job(job_id, x_user_id)
    if job.status != DONE:
        raise HTTPException(409, f"Job chưa xong (status={job.status}). Poll {API_PREFIX}/tts/{job_id}.")
    if not job.audio_url:
        raise HTTPException(409, "Job xong nhưng không có audio.")

    # Audio nằm ở storage. Với local → đọc thẳng qua key; với R2 → tải qua URL.
    from .storage import get_storage
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
