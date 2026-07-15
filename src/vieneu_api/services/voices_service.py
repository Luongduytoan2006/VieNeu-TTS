"""Voice CRUD: clone a voice from an uploaded audio clip and manage the catalog.

Enrolling = analyze the clip (denoise → speaker embedding + reference codes) via
the SDK's ``add_voice``, persist the clip + a DB row, and re-save the custom
voices JSON so they survive a restart. Preset (built-in) voices are read-only.
"""
from __future__ import annotations

import logging
from typing import List

from ..config import settings
from ..db import session_scope
from ..models import Voice
from ..schemas import DEFAULT_STYLE, VoiceInfo
from .tts_service import service

logger = logging.getLogger("Vieneu.API.voices")


class VoiceExistsError(Exception):
    pass


class VoiceNotFoundError(Exception):
    pass


class PresetVoiceReadonlyError(Exception):
    pass


def list_voices() -> List[VoiceInfo]:
    """All voices the hot model can use, marking which are custom (from DB)."""
    default_v = service.default_voice()
    custom_ids = _custom_ids()
    out: List[VoiceInfo] = []
    for label, vid in service.list_voices():
        meta = service.voice_meta(vid)
        out.append(VoiceInfo(
            id=vid, label=label, gender=meta.get("gender") or None,
            style=meta.get("style", DEFAULT_STYLE),
            source="custom" if vid in custom_ids else "preset",
            is_default=(vid == default_v),
        ))
    return out


def get_voice(voice_id: str) -> VoiceInfo:
    for v in list_voices():
        if v.id == voice_id:
            return v
    raise VoiceNotFoundError(voice_id)


def add_voice(name: str, ref_audio_path: str, *, description: str = "",
              gender: str = "", style: str = DEFAULT_STYLE, denoise: bool = True) -> VoiceInfo:
    name = (name or "").strip()
    if not name:
        raise ValueError("Tên giọng không được để trống.")
    if service.has_voice(name):
        raise VoiceExistsError(name)

    # Analyze + register in the hot model.
    service.enroll_voice(name, ref_audio_path, denoise=denoise,
                         description=description, gender=gender, style=style)
    # Persist: DB row + re-save custom voices JSON (embeddings + codes).
    with session_scope() as s:
        s.add(Voice(id=name, description=description, gender=gender or None,
                    style=style, source="custom", ref_audio_path=ref_audio_path))
    _persist_custom()
    logger.info("➕ enrolled voice '%s'", name)
    return get_voice(name)


def delete_voice(voice_id: str) -> None:
    if voice_id not in _custom_ids():
        # Either unknown, or a built-in preset (read-only).
        if service.has_voice(voice_id):
            raise PresetVoiceReadonlyError(voice_id)
        raise VoiceNotFoundError(voice_id)
    service.remove_voice(voice_id)
    with session_scope() as s:
        row = s.get(Voice, voice_id)
        if row is not None:
            s.delete(row)
    _persist_custom()
    logger.info("🗑️ deleted voice '%s'", voice_id)


# ── Persistence ──────────────────────────────────────────────────────────────
def _custom_ids() -> set[str]:
    with session_scope() as s:
        return {v.id for v in s.query(Voice.id).all()}


def _persist_custom() -> None:
    """Write ONLY the custom voices to the custom-voices JSON in the data volume.

    Kept separate from the built-in assets file so a rebuild never clobbers
    enrolled voices (they live in the mounted data dir).
    """
    import json
    import numpy as np

    tts = service.tts
    if tts is None:
        return
    custom = _custom_ids()
    presets = {}
    for name in custom:
        v = tts._preset_voices.get(name)
        if not v:
            continue
        emb = v.get("speaker_emb")
        codes = v.get("codes")
        presets[name] = {
            "description": v.get("description", ""),
            "gender": v.get("gender", ""),
            "style": v.get("style", DEFAULT_STYLE),
            "speaker_emb": [round(float(x), 6) for x in np.asarray(emb).reshape(-1)] if emb is not None else None,
            "codes": np.asarray(codes, dtype=int).tolist() if codes is not None else None,
        }
    data = {"meta": {"note": "custom enrolled voices"}, "presets": presets}
    settings.CUSTOM_VOICES_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def restore_custom_voices() -> int:
    """On startup: load persisted custom voices into the hot model."""
    return service.load_custom_voices(settings.CUSTOM_VOICES_PATH)
