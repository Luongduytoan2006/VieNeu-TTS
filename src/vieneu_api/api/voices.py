"""Voice CRUD — clone from an uploaded audio clip.

Works in both modes: local enrolls in the in-process model; remote forwards the
upload to the GPU model-server (which owns the model + storage).
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import settings
from ..schemas import DEFAULT_STYLE, ErrorResponse, VoiceInfo, VoicesResponse
from ..services.tts_service import service
from ..services import catalog_provider as catalog
from ..services import voices_service as vs

router = APIRouter(prefix="/voices", tags=["voices"])

_ALLOWED_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@router.get("", response_model=VoicesResponse, summary="Danh sách giọng (preset + custom)")
def list_voices() -> VoicesResponse:
    if not catalog.ready():
        raise HTTPException(503, "Model chưa sẵn sàng.")
    if catalog.is_remote():
        default_v = catalog.default_voice()
        items = []
        for label, vid in catalog.list_voice_tuples():
            m = catalog.voice_meta(vid)
            items.append(VoiceInfo(id=vid, label=label, gender=m.get("gender") or None,
                                   style=m.get("style", DEFAULT_STYLE), source="preset",
                                   is_default=(vid == default_v)))
        return VoicesResponse(count=len(items), default_voice=default_v, voices=items)
    items = vs.list_voices()
    return VoicesResponse(count=len(items), default_voice=service.default_voice(), voices=items)


@router.get("/{voice_id}", response_model=VoiceInfo,
            responses={404: {"model": ErrorResponse}}, summary="Chi tiết 1 giọng")
def get_voice(voice_id: str) -> VoiceInfo:
    if catalog.is_remote():
        if not catalog.has_voice(voice_id):
            raise HTTPException(404, f"Voice '{voice_id}' không tồn tại.")
        m = catalog.voice_meta(voice_id)
        label = next((l for l, v in catalog.list_voice_tuples() if v == voice_id), voice_id)
        return VoiceInfo(id=voice_id, label=label, gender=m.get("gender") or None,
                         style=m.get("style", DEFAULT_STYLE), source="preset",
                         is_default=(voice_id == catalog.default_voice()))
    try:
        return vs.get_voice(voice_id)
    except vs.VoiceNotFoundError:
        raise HTTPException(404, f"Voice '{voice_id}' không tồn tại.")


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
    """Gửi 1 file audio → denoise + trích đặc trưng giọng → thêm vào list.

    Local: enroll in-process. Remote: forward tới model-server (GPU) enroll.
    """
    if not catalog.ready():
        raise HTTPException(503, "Model chưa sẵn sàng.")
    ext = ("." + audio.filename.rsplit(".", 1)[-1].lower()) if audio.filename and "." in audio.filename else ".wav"
    if ext not in _ALLOWED_EXT:
        raise HTTPException(422, f"Định dạng '{ext}' không hỗ trợ. Cho phép: {sorted(_ALLOWED_EXT)}")
    data = await audio.read()
    if not data:
        raise HTTPException(422, "File audio rỗng.")

    if catalog.is_remote():
        from ..services.model_client import client, ModelClientError
        try:
            client.enroll_voice(name, data, audio.filename or "ref.wav",
                                description=description, gender=gender, style=style, denoise=denoise)
        except ModelClientError as e:
            raise HTTPException(e.status if e.status in (409, 422, 500, 503) else 502, e.detail)
        catalog.invalidate()
        return VoiceInfo(id=name, label=name, gender=gender or None, style=style,
                         source="custom", is_default=False)

    # local
    dest = settings.VOICE_UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
    dest.write_bytes(data)
    try:
        return vs.add_voice(name, str(dest), description=description, gender=gender,
                            style=style, denoise=denoise)
    except vs.VoiceExistsError:
        dest.unlink(missing_ok=True)
        raise HTTPException(409, f"Voice '{name}' đã tồn tại. Xóa trước hoặc đổi tên.")
    except ValueError as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, str(e))
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(500, f"Lỗi phân tích giọng: {type(e).__name__}: {e}")


@router.delete("/{voice_id}", status_code=204,
               responses={404: {"model": ErrorResponse}, 403: {"model": ErrorResponse}},
               summary="Xóa giọng custom (không xóa được preset)")
def delete_voice(voice_id: str):
    if catalog.is_remote():
        from ..services.model_client import client, ModelClientError
        try:
            client.delete_voice(voice_id)
        except ModelClientError as e:
            raise HTTPException(e.status if e.status in (404,) else 502, e.detail)
        catalog.invalidate()
        return None
    try:
        vs.delete_voice(voice_id)
    except vs.PresetVoiceReadonlyError:
        raise HTTPException(403, f"'{voice_id}' là giọng mặc định (preset), không thể xóa.")
    except vs.VoiceNotFoundError:
        raise HTTPException(404, f"Voice '{voice_id}' không tồn tại.")
    return None
