"""voices — CRUD giọng: preset (RAM, dùng chung) + custom (DB, theo user).

* preset — 10 giọng hệ thống SDK nạp sẵn trong RAM (``engine``), bất biến, chung
  cho mọi user, KHÔNG lưu DB.
* custom — user clone qua ``POST /voices``: trích emb+codes → lưu DB theo ``user_ref``.

GET /voices của 1 user = preset (chung) + custom của chính user đó. Mọi thao tác
ghi/đọc custom đều kèm ``user_ref`` (khóa (user_ref, id) → 2 user trùng tên OK).
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List, Optional

from ..config import settings
from ..engine import engine
from ..repositories import voices_repo as repo
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
def _to_info(rec: dict, default_id: Optional[str]) -> VoiceInfo:
    vid = rec["id"]
    label = f"{vid} — {rec['description']}" if rec.get("description") else vid
    return VoiceInfo(
        id=vid, label=label, gender=rec.get("gender") or None,
        style=rec.get("style", DEFAULT_STYLE),
        source=rec.get("source", "custom"),
        is_default=(rec.get("source") == "preset" and vid == default_id),
    )


def list_voices(user_ref: str = "default") -> VoicesResponse:
    """Preset (RAM, chung) + custom của user (DB)."""
    dv = engine.default_preset_id()
    presets = [_to_info(r, dv) for r in engine.list_preset_records()]
    customs = [_to_info(r, dv) for r in repo.list_for_user(user_ref)]
    items = presets + customs
    return VoicesResponse(count=len(items), default_voice=dv, voices=items)


def get_voice(voice_id: str, user_ref: str = "default") -> Optional[VoiceInfo]:
    dv = engine.default_preset_id()
    rec = engine.get_preset_record(voice_id) or repo.get(user_ref, voice_id)
    return _to_info(rec, dv) if rec is not None else None


# ── Write ─────────────────────────────────────────────────────────────────────
def enroll(name: str, audio_bytes: bytes, filename: str, *, user_ref: str = "default",
           description: str = "", gender: str = "", style: str = DEFAULT_STYLE,
           denoise: bool = True) -> VoiceInfo:
    """Clone 1 giọng mới từ audio mẫu (3–8s) → trích đặc trưng → LƯU DB theo user."""
    if not engine.loaded:
        raise VoiceError(503, "Model chưa sẵn sàng.")
    if engine.has_preset(name):
        raise VoiceError(409, f"'{name}' trùng tên giọng preset hệ thống. Đặt tên khác.")
    if repo.exists(user_ref, name):
        raise VoiceError(409, f"Voice '{name}' đã tồn tại.")

    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if filename and "." in filename else ".wav"
    if ext not in ALLOWED_EXT:
        raise VoiceError(422, f"Định dạng '{ext}' không hỗ trợ. Cho phép: {sorted(ALLOWED_EXT)}")
    if not audio_bytes:
        raise VoiceError(422, "File audio rỗng.")

    dest = _UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
    dest.write_bytes(audio_bytes)
    try:
        emb, codes = engine.extract_reference(str(dest), denoise=denoise)
    except Exception as e:
        raise VoiceError(500, f"Lỗi phân tích giọng: {type(e).__name__}: {e}")
    finally:
        dest.unlink(missing_ok=True)

    repo.upsert(user_ref, name, speaker_emb=emb, codes=codes, name=name,
                description=description, gender=gender, default_style=style)
    logger.info("➕ enrolled voice '%s' → DB user=%s", name, user_ref)
    return VoiceInfo(id=name, label=name, gender=gender or None, style=style,
                     source="custom", is_default=False)


def delete(voice_id: str, user_ref: str = "default") -> None:
    if engine.has_preset(voice_id):
        raise VoiceError(409, f"Voice '{voice_id}' là preset hệ thống, không thể xóa.")
    if not repo.exists(user_ref, voice_id):
        raise VoiceError(404, f"Voice '{voice_id}' không tồn tại.")
    repo.delete(user_ref, voice_id)
    logger.info("🗑️ deleted voice '%s' user=%s", voice_id, user_ref)


__all__ = ["list_voices", "get_voice", "enroll", "delete", "ALLOWED_EXT", "VoiceError"]
