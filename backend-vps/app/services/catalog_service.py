"""Read-only catalog: voices, styles, modes, readiness — sourced from the
model-server (cached briefly). The API layer calls these instead of hitting the
model-server directly, so caching/shape lives in one place.
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional, Tuple

from ..schemas import (
    DEFAULT_STYLE, STYLE_CHOICES, ModeInfo, ModesResponse, StyleInfo, StylesResponse,
    VoiceInfo, VoicesResponse,
)
from .model_client import client

_cache: dict = {"voices": None, "ts": 0.0}
_lock = threading.Lock()
_TTL = 5.0


# ── readiness ────────────────────────────────────────────────────────────────
def ready() -> bool:
    try:
        return bool(client.health().get("model_loaded"))
    except Exception:
        return False


def model_health() -> dict:
    """Raw model-server health (or an error-shaped dict)."""
    try:
        return client.health()
    except Exception as e:
        return {"status": "unreachable", "model_loaded": False, "error": str(e)}


# ── voices ───────────────────────────────────────────────────────────────────
def _remote_voices() -> List[dict]:
    now = time.time()
    with _lock:
        if _cache["voices"] is not None and now - _cache["ts"] < _TTL:
            return _cache["voices"]
    data = client.voices()
    voices = data.get("voices", [])
    with _lock:
        _cache["voices"] = voices
        _cache["ts"] = now
    return voices


def invalidate() -> None:
    with _lock:
        _cache["voices"] = None


def default_voice() -> Optional[str]:
    for v in _remote_voices():
        if v.get("is_default"):
            return v["id"]
    return None


def _voice_tuples() -> List[Tuple[str, str]]:
    return [(v.get("label", v["id"]), v["id"]) for v in _remote_voices()]


def has_voice(voice_id: str) -> bool:
    return any(v["id"] == voice_id for v in _remote_voices())


def _voice_meta(voice_id: str) -> dict:
    for v in _remote_voices():
        if v["id"] == voice_id:
            return v
    return {}


def list_voices() -> VoicesResponse:
    dv = default_voice()
    items: List[VoiceInfo] = []
    for v in _remote_voices():
        items.append(VoiceInfo(
            id=v["id"], label=v.get("label", v["id"]), gender=v.get("gender") or None,
            style=v.get("style", DEFAULT_STYLE),
            source=v.get("source", "preset"), is_default=(v["id"] == dv),
        ))
    return VoicesResponse(count=len(items), default_voice=dv, voices=items)


def get_voice(voice_id: str) -> Optional[VoiceInfo]:
    v = _voice_meta(voice_id)
    if not v:
        return None
    return VoiceInfo(
        id=v["id"], label=v.get("label", v["id"]), gender=v.get("gender") or None,
        style=v.get("style", DEFAULT_STYLE), source=v.get("source", "preset"),
        is_default=(v["id"] == default_voice()),
    )


# ── styles / modes ───────────────────────────────────────────────────────────
def list_styles() -> StylesResponse:
    items = [StyleInfo(id=sid, label=label, is_default=(sid == DEFAULT_STYLE))
             for sid, label in STYLE_CHOICES.items()]
    return StylesResponse(count=len(items), default_style=DEFAULT_STYLE, styles=items)


def list_modes() -> ModesResponse:
    h = model_health()
    backend = h.get("backend") or "unknown"
    device = h.get("device")
    items = [
        ModeInfo(id="v3turbo", label="VieNeu-TTS v3 Turbo (48kHz)",
                 description="GPU dùng PyTorch, CPU dùng ONNX. Voice cloning, tag cảm xúc.",
                 available=True),
        ModeInfo(id="pytorch", label="PyTorch (GPU)",
                 description="Engine PyTorch cho GPU CUDA.", available=(device == "cuda")),
        ModeInfo(id="onnx", label="ONNX (CPU)",
                 description="Engine ONNX torch-free cho CPU.", available=(device == "cpu")),
    ]
    return ModesResponse(count=len(items), active_mode=backend, modes=items)
