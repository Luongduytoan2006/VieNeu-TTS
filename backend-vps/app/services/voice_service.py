"""Voice CRUD — forwards enroll/delete to the GPU model-server (which owns the
model + storage), then invalidates the catalog cache. backend-vps holds no model,
so there is nothing to persist locally beyond what the model-server already keeps.
"""
from __future__ import annotations

import logging

from ..schemas import DEFAULT_STYLE, VoiceInfo
from . import catalog_service
from .model_client import ModelClientError, client

logger = logging.getLogger("Vieneu.VPS.voice_service")

ALLOWED_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def enroll(name: str, audio_bytes: bytes, filename: str, *, description: str = "",
           gender: str = "", style: str = DEFAULT_STYLE, denoise: bool = True) -> VoiceInfo:
    """Forward the uploaded clip to the model-server for cloning."""
    client.enroll_voice(name, audio_bytes, filename, description=description,
                        gender=gender, style=style, denoise=denoise)
    catalog_service.invalidate()
    logger.info("➕ enrolled voice '%s' (forwarded to model-server)", name)
    return VoiceInfo(id=name, label=name, gender=gender or None, style=style,
                     source="custom", is_default=False)


def delete(voice_id: str) -> None:
    client.delete_voice(voice_id)
    catalog_service.invalidate()
    logger.info("🗑️ deleted voice '%s' (forwarded to model-server)", voice_id)


__all__ = ["enroll", "delete", "ALLOWED_EXT", "ModelClientError"]
