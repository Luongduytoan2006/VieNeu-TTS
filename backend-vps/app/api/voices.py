"""Voice CRUD — list/get, plus clone-from-upload and delete (forwarded to GPU)."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..schemas import DEFAULT_STYLE, ErrorResponse, VoiceInfo, VoicesResponse
from ..services import catalog_service
from ..services import voice_service
from ..services.model_client import ModelClientError

router = APIRouter(prefix="/voices", tags=["voices"])


@router.get("", response_model=VoicesResponse, summary="Danh sách giọng (preset + custom)")
def list_voices() -> VoicesResponse:
    if not catalog_service.ready():
        raise HTTPException(503, "Model chưa sẵn sàng.")
    return catalog_service.list_voices()


@router.get("/{voice_id}", response_model=VoiceInfo,
            responses={404: {"model": ErrorResponse}}, summary="Chi tiết 1 giọng")
def get_voice(voice_id: str) -> VoiceInfo:
    if not catalog_service.ready():
        raise HTTPException(503, "Model chưa sẵn sàng.")
    v = catalog_service.get_voice(voice_id)
    if v is None:
        raise HTTPException(404, f"Voice '{voice_id}' không tồn tại.")
    return v


@router.post("", response_model=VoiceInfo, status_code=201,
             responses={409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
             summary="Nạp giọng mới từ audio (voice cloning)")
async def create_voice(
    name: str = Form(..., description="Tên/ID giọng mới."),
    audio: UploadFile = File(..., description="File audio mẫu 3–8s (wav/mp3/...)."),
    description: str = Form(""),
    gender: str = Form(""),
    style: str = Form(DEFAULT_STYLE),
    denoise: bool = Form(True),
) -> VoiceInfo:
    """Gửi 1 file audio → model-server denoise + trích đặc trưng giọng → thêm vào list."""
    if not catalog_service.ready():
        raise HTTPException(503, "Model chưa sẵn sàng.")
    ext = ("." + audio.filename.rsplit(".", 1)[-1].lower()) if audio.filename and "." in audio.filename else ".wav"
    if ext not in voice_service.ALLOWED_EXT:
        raise HTTPException(422, f"Định dạng '{ext}' không hỗ trợ. Cho phép: {sorted(voice_service.ALLOWED_EXT)}")
    data = await audio.read()
    if not data:
        raise HTTPException(422, "File audio rỗng.")
    try:
        return voice_service.enroll(name, data, audio.filename or "ref.wav",
                                    description=description, gender=gender,
                                    style=style, denoise=denoise)
    except ModelClientError as e:
        raise HTTPException(e.status if e.status in (409, 422, 500, 503) else 502, e.detail)


@router.delete("/{voice_id}", status_code=204,
               responses={404: {"model": ErrorResponse}, 403: {"model": ErrorResponse}},
               summary="Xóa giọng custom (không xóa được preset)")
def delete_voice(voice_id: str):
    try:
        voice_service.delete(voice_id)
    except ModelClientError as e:
        raise HTTPException(e.status if e.status in (403, 404) else 502, e.detail)
    return None
