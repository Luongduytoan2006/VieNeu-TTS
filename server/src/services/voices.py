"""voices — CRUD giọng (list/get + clone-from-upload + delete), DB-backed.

Giọng nằm ở DB (``repositories/voices_repo``) — nguồn sự thật, sống qua restart.
Enroll: ghi audio tạm → ``engine.extract_reference`` (denoise + trích emb+codes)
→ LƯU DB (source=custom). Đọc/list/delete đều qua repo, KHÔNG tra RAM engine.
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
        is_default=(vid == default_id),
    )


def list_voices() -> VoicesResponse:
    dv = repo.default_voice_id()
    items: List[VoiceInfo] = [_to_info(r, dv) for r in repo.list_all()]
    return VoicesResponse(count=len(items), default_voice=dv, voices=items)


def get_voice(voice_id: str) -> Optional[VoiceInfo]:
    rec = repo.get(voice_id)
    if rec is None:
        return None
    return _to_info(rec, repo.default_voice_id())


# ── Write ─────────────────────────────────────────────────────────────────────
def enroll(name: str, audio_bytes: bytes, filename: str, *, description: str = "",
           gender: str = "", style: str = DEFAULT_STYLE, denoise: bool = True) -> VoiceInfo:
    """Clone 1 giọng mới từ audio mẫu (3–8s) → trích đặc trưng → LƯU DB."""
    if not engine.loaded:
        raise VoiceError(503, "Model chưa sẵn sàng.")
    if repo.exists(name):
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

    repo.upsert(name, speaker_emb=emb, codes=codes, name=name, description=description,
                gender=gender, default_style=style, source="custom")
    logger.info("➕ enrolled voice '%s' → DB (source=custom)", name)
    return VoiceInfo(id=name, label=name, gender=gender or None, style=style,
                     source="custom", is_default=False)


def delete(voice_id: str) -> None:
    rec = repo.get(voice_id)
    if rec is None:
        raise VoiceError(404, f"Voice '{voice_id}' không tồn tại.")
    if rec.get("source") == "preset":
        raise VoiceError(409, f"Voice '{voice_id}' là preset, không thể xóa.")
    repo.delete(voice_id)
    logger.info("🗑️ deleted voice '%s' khỏi DB", voice_id)


def seed_presets() -> int:
    """Đẩy 10 giọng preset (SDK nạp sẵn trong RAM) vào DB — gọi 1 lần lúc boot.

    Idempotent: giọng preset đã có thì cập nhật lại (giữ nguyên custom của user).
    Sau bước này DB là nguồn sự thật DUY NHẤT cho mọi giọng. Trả số giọng seed.
    """
    seeds = engine.preset_seed_records()
    n = 0
    for s in seeds:
        repo.upsert(
            s["id"], speaker_emb=s.get("speaker_emb"), codes=s.get("codes"),
            name=s["id"], description=s.get("description", ""),
            gender=s.get("gender", ""), default_style=s.get("style", DEFAULT_STYLE),
            source="preset", is_default=s.get("is_default", False),
        )
        n += 1
    if n:
        logger.info("🌱 seed %d giọng preset vào DB (tổng %d giọng)", n, repo.count())
    return n


__all__ = ["list_voices", "get_voice", "enroll", "delete", "seed_presets",
           "ALLOWED_EXT", "VoiceError"]
