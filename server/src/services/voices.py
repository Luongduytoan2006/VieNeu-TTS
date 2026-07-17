"""voices — CRUD giọng (list/get + clone-from-upload + delete), in-process.

Gọi thẳng vào ``src.engine`` (gốc tác giả). Enroll: ghi file audio tạm → engine
denoise + trích đặc trưng → thêm vào danh sách preset của model đang chạy.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List, Optional

from ..config import settings
from ..engine import engine
from ..schemas import DEFAULT_STYLE, VoiceInfo, VoicesResponse

logger = logging.getLogger("Vieneu.Voices")

ALLOWED_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
_UPLOAD_DIR = Path(settings.STORAGE_LOCAL_DIR) / "uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class VoiceError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


# ── Read ──────────────────────────────────────────────────────────────────────
def _to_info(vid: str, label: str, meta: dict, default: Optional[str]) -> VoiceInfo:
    return VoiceInfo(
        id=vid, label=label, gender=meta.get("gender") or None,
        style=meta.get("style", DEFAULT_STYLE),
        source="custom" if meta.get("_custom") else "preset",
        is_default=(vid == default),
    )


def list_voices() -> VoicesResponse:
    dv = engine.default_voice()
    items: List[VoiceInfo] = []
    for label, vid in engine.list_voices():
        items.append(_to_info(vid, label, engine.voice_meta(vid), dv))
    return VoicesResponse(count=len(items), default_voice=dv, voices=items)


def get_voice(voice_id: str) -> Optional[VoiceInfo]:
    if not engine.has_voice(voice_id):
        return None
    meta = engine.voice_meta(voice_id)
    label = f"{voice_id} — {meta.get('description')}" if meta.get("description") else voice_id
    return _to_info(voice_id, label, meta, engine.default_voice())


# ── Write ─────────────────────────────────────────────────────────────────────
def enroll(name: str, audio_bytes: bytes, filename: str, *, description: str = "",
           gender: str = "", style: str = DEFAULT_STYLE, denoise: bool = True) -> VoiceInfo:
    """Clone 1 giọng mới từ audio mẫu (3–8s)."""
    if not engine.loaded:
        raise VoiceError(503, "Model chưa sẵn sàng.")
    if engine.has_voice(name):
        raise VoiceError(409, f"Voice '{name}' đã tồn tại.")

    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if filename and "." in filename else ".wav"
    if ext not in ALLOWED_EXT:
        raise VoiceError(422, f"Định dạng '{ext}' không hỗ trợ. Cho phép: {sorted(ALLOWED_EXT)}")
    if not audio_bytes:
        raise VoiceError(422, "File audio rỗng.")

    dest = _UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
    dest.write_bytes(audio_bytes)
    try:
        engine.enroll(name, str(dest), description=description, gender=gender,
                      style=style, denoise=denoise)
    except Exception as e:
        raise VoiceError(500, f"Lỗi phân tích giọng: {type(e).__name__}: {e}")
    finally:
        dest.unlink(missing_ok=True)

    logger.info("➕ enrolled voice '%s' (in-process)", name)
    return VoiceInfo(id=name, label=name, gender=gender or None, style=style,
                     source="custom", is_default=False)


def delete(voice_id: str) -> None:
    if not engine.has_voice(voice_id):
        raise VoiceError(404, f"Voice '{voice_id}' không tồn tại.")
    engine.remove(voice_id)
    logger.info("🗑️ deleted voice '%s'", voice_id)


__all__ = ["list_voices", "get_voice", "enroll", "delete", "ALLOWED_EXT", "VoiceError"]
