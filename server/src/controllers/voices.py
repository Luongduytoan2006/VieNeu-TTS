"""Voice routes — list/detail/enroll/delete voices."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile

from ..config import settings
from ..schemas import DEFAULT_STYLE, ErrorResponse, VoiceInfo, VoicesResponse
from ..services import catalog as catalog_svc
from ..services import voices as voice_svc

router = APIRouter(prefix=settings.API_PREFIX, tags=["voices"])

_UID = Header(default=None, alias="X-User-Id", description="ID người dùng (module ngoài truyền vào).")


def _user(x_user_id: Optional[str]) -> str:
    uid = (x_user_id or "").strip()
    return uid or "default"


@router.get("/voices", response_model=VoicesResponse,
            summary="GET /api/v1/voices — Danh sách giọng (preset + custom của bạn)")
def list_voices(x_user_id: Optional[str] = _UID) -> VoicesResponse:
    if not catalog_svc.ready():
        raise HTTPException(503, "Model chưa sẵn sàng.")
    return voice_svc.list_voices(user_ref=_user(x_user_id))


@router.get("/voices/{voice_id}", response_model=VoiceInfo,
            responses={404: {"model": ErrorResponse}},
            summary="GET /api/v1/voices/{voice_id} — Chi tiết 1 giọng")
def get_voice(voice_id: str, x_user_id: Optional[str] = _UID) -> VoiceInfo:
    if not catalog_svc.ready():
        raise HTTPException(503, "Model chưa sẵn sàng.")
    v = voice_svc.get_voice(voice_id, user_ref=_user(x_user_id))
    if v is None:
        raise HTTPException(404, f"Voice '{voice_id}' không tồn tại.")
    return v


@router.post("/voices", response_model=VoiceInfo, status_code=201,
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


@router.delete("/voices/{voice_id}", status_code=204,
               responses={404: {"model": ErrorResponse}},
               summary="DELETE /api/v1/voices/{voice_id} — Xóa giọng custom")
def delete_voice(voice_id: str, x_user_id: Optional[str] = _UID):
    try:
        voice_svc.delete(voice_id, user_ref=_user(x_user_id))
    except voice_svc.VoiceError as e:
        raise HTTPException(e.status, e.detail)
    return None
