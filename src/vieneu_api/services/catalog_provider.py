"""Mode-agnostic view of readiness + voice catalog for the routers.

In local mode this reads the in-process model (tts_service); in remote mode it
reads the model-server over HTTP (cached briefly). Routers call these helpers
instead of touching the model directly, so they work in both modes unchanged.
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional, Tuple

from ..config import settings
from .tts_service import service

_cache: dict = {"voices": None, "ts": 0.0}
_lock = threading.Lock()
_TTL = 5.0


def is_remote() -> bool:
    return settings.MODE == "remote"


def ready() -> bool:
    if is_remote():
        try:
            from .model_client import client
            return bool(client.health().get("model_loaded"))
        except Exception:
            return False
    return service.loaded


def _remote_voices() -> List[dict]:
    now = time.time()
    with _lock:
        if _cache["voices"] is not None and now - _cache["ts"] < _TTL:
            return _cache["voices"]
    from .model_client import client
    data = client.voices()
    voices = data.get("voices", [])
    with _lock:
        _cache["voices"] = voices
        _cache["ts"] = now
    return voices


def default_voice() -> Optional[str]:
    if is_remote():
        for v in _remote_voices():
            if v.get("is_default"):
                return v["id"]
        return None
    return service.default_voice()


def list_voice_tuples() -> List[Tuple[str, str]]:
    """[(label, id)] — matches the in-process SDK shape."""
    if is_remote():
        return [(v.get("label", v["id"]), v["id"]) for v in _remote_voices()]
    return service.list_voices()


def voice_meta(voice_id: str) -> dict:
    if is_remote():
        for v in _remote_voices():
            if v["id"] == voice_id:
                return {"gender": v.get("gender"), "style": v.get("style")}
        return {}
    return service.voice_meta(voice_id)


def has_voice(voice_id: str) -> bool:
    if is_remote():
        return any(v["id"] == voice_id for v in _remote_voices())
    return service.has_voice(voice_id)


def invalidate() -> None:
    with _lock:
        _cache["voices"] = None
