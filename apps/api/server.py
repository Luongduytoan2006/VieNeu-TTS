"""FastAPI app: UI + REST API + Swagger on a SINGLE port.

Routes
------
- ``GET  /``            → the Gradio web UI (mounted), same as before.
- ``GET  /docs``        → Swagger UI (auto-generated from the pydantic schemas).
- ``GET  /api/v1/health``   → readiness + architecture summary.
- ``GET  /api/v1/voices``   → list preset voices (+ ids).
- ``GET  /api/v1/styles``   → list reading styles (+ ids).
- ``POST /api/v1/tts``      → synthesize text → returns a WAV file.

Why one port: the Proxmox CT only exposes a single port (7862) through nginx, so
the UI and the API must live behind the same server. We build a FastAPI app,
add the REST routes, then mount the Gradio Blocks at ``/`` with
``gradio.mount_gradio_app``. The model is loaded once at startup and shared.

Run locally:  ``uv run vieneu-api``  (honors PORT / HOST env vars).
"""
from __future__ import annotations

import io
import logging
import os
import wave
from typing import Optional
from urllib.parse import quote

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse

from .model_manager import manager
from .schemas import (
    DEFAULT_STYLE,
    STYLE_CHOICES,
    ErrorResponse,
    HealthResponse,
    StyleInfo,
    StylesResponse,
    TTSRequest,
    VoiceInfo,
    VoicesResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("Vieneu.API")

API_PREFIX = "/api/v1"


class ForceHttpsSchemeMiddleware:
    """Rewrite the ``x-forwarded-proto`` header to ``https`` so the mounted Gradio
    UI builds https:// asset URLs when we're served over HTTPS behind a proxy.

    Cloudflare terminates TLS and calls nginx over plain HTTP, so nginx sends
    ``X-Forwarded-Proto: http`` to us and the real https is lost. Gradio builds its
    root_url from this header (gradio/route_utils.py: only upgrades to https when
    ``x-forwarded-proto == "https"``), so it hands the browser http:// audio URLs
    on an https page and the browser blocks them as mixed content (player stuck at
    0:00, download fails). We set the header to https here.

    Enabled only when ``VIENEU_FORCE_HTTPS=1`` (server compose), so local http dev
    is unaffected. We overwrite the header (Cloudflare's edge already spoke https to
    the user); set VIENEU_FORCE_HTTPS=0 if you ever serve this over plain http.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            scope = dict(scope)
            headers = [
                (n, v) for (n, v) in (scope.get("headers") or [])
                if n.lower() not in (b"x-forwarded-proto", b"x-forwarded-scheme")
            ]
            headers.append((b"x-forwarded-proto", b"https"))
            headers.append((b"x-forwarded-scheme", b"https"))
            scope["headers"] = headers
            scope["scheme"] = "https"
        await self.app(scope, receive, send)

app = FastAPI(
    title="VieNeu-TTS API",
    version="1.0.0",
    description=(
        "REST API cho VieNeu-TTS v3 Turbo (48 kHz, tiếng Việt/Anh). "
        "Gọi thẳng vào SDK in-process — text vào, file WAV ra. "
        "Không cần token/đăng nhập."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# When served over HTTPS behind Cloudflare+nginx, force the request scheme to https
# so Gradio emits https:// asset URLs (see ForceHttpsSchemeMiddleware). Off for
# plain local http unless VIENEU_FORCE_HTTPS=1.
if os.getenv("VIENEU_FORCE_HTTPS", "0") == "1":
    app.add_middleware(ForceHttpsSchemeMiddleware)
    logger.info("🔒 ForceHttpsScheme middleware enabled (VIENEU_FORCE_HTTPS=1).")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    """Encode a float32 [-1,1] mono waveform as a 16-bit PCM WAV byte string."""
    if wav is None or len(wav) == 0:
        raise HTTPException(status_code=500, detail="Không sinh được audio nào (wav rỗng).")
    pcm16 = (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


# ── Lifecycle ────────────────────────────────────────────────────────────────
@app.on_event("startup")
def _startup() -> None:
    """Load the model at boot so the first request isn't cold.

    If loading fails we DON'T crash the server — /health will report the error
    so the box stays reachable for debugging.
    """
    if os.getenv("VIENEU_API_EAGER_LOAD", "1") == "1":
        try:
            manager.load()
        except Exception:
            logger.error("Startup model load failed; server stays up, /health will show error.")


# ── REST endpoints ───────────────────────────────────────────────────────────
@app.get(
    f"{API_PREFIX}/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Kiểm tra sức khỏe + kiến trúc",
)
def health() -> HealthResponse:
    """Trả về trạng thái tổng thể: model đã nạp chưa, backend/device, số giọng."""
    loaded = manager.loaded
    status = "ok" if loaded else ("error" if manager.load_error else "loading")
    return HealthResponse(
        status=status,
        model_loaded=loaded,
        backend=manager.backend,
        device=manager.device,
        backbone_repo=manager.backbone_repo,
        sample_rate=manager.sample_rate,
        num_voices=len(manager.list_voices()),
        ui_mounted=bool(getattr(app.state, "ui_mounted", False)),
    )


@app.get(
    f"{API_PREFIX}/voices",
    response_model=VoicesResponse,
    tags=["catalog"],
    summary="Danh sách giọng preset",
)
def voices() -> VoicesResponse:
    """Liệt kê toàn bộ giọng mẫu kèm ID (dùng ID này ở trường `voice` khi tạo audio)."""
    if not manager.loaded:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng. Xem GET /api/v1/health.")
    default_v = manager.default_voice()
    items = []
    for label, vid in manager.list_voices():
        meta = manager.voice_meta(vid)
        items.append(
            VoiceInfo(
                id=vid,
                label=label,
                gender=meta.get("gender") or None,
                is_default=(vid == default_v),
            )
        )
    return VoicesResponse(count=len(items), default_voice=default_v, voices=items)


@app.get(
    f"{API_PREFIX}/styles",
    response_model=StylesResponse,
    tags=["catalog"],
    summary="Danh sách phong cách đọc",
)
def styles() -> StylesResponse:
    """Ba phong cách: tự nhiên / tin tức / kể chuyện."""
    items = [
        StyleInfo(id=sid, label=label, is_default=(sid == DEFAULT_STYLE))
        for sid, label in STYLE_CHOICES.items()
    ]
    return StylesResponse(count=len(items), default_style=DEFAULT_STYLE, styles=items)


@app.post(
    f"{API_PREFIX}/tts",
    tags=["synthesis"],
    summary="Tạo âm thanh từ văn bản (trả file WAV)",
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": "File WAV 48 kHz. Metadata ở các header X-* (X-Status, X-Duration-Sec, "
            "X-Elapsed-Sec, X-Realtime-Factor, X-Voice, X-Style).",
        },
        422: {"model": ErrorResponse, "description": "Tham số không hợp lệ."},
        503: {"model": ErrorResponse, "description": "Model chưa sẵn sàng."},
    },
)
def tts(req: TTSRequest) -> Response:
    """Tổng hợp giọng nói. Body JSON; trả về **file WAV** + trạng thái ở HTTP headers.

    Ví dụ trạng thái (đọc ở header `X-Status`):
    ``✅ Hoàn tất! (1 giọng, 3.2s, 48000Hz, 2.10x realtime)``
    """
    if not manager.loaded:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng. Xem GET /api/v1/health.")

    # Validate voice id early → clear 422 instead of a 500 deep in the engine.
    voice = req.voice
    if voice:
        valid_ids = {vid for _, vid in manager.list_voices()}
        if voice not in valid_ids:
            raise HTTPException(
                status_code=422,
                detail=f"Voice '{voice}' không tồn tại. Xem GET {API_PREFIX}/voices.",
            )

    style = req.validated_style()
    try:
        wav, sr, elapsed = manager.synthesize(
            text=req.text,
            voice=voice,
            style=style,
            temperature=req.temperature,
            max_chars=req.max_chars,
            apply_watermark=req.apply_watermark,
        )
    except Exception as e:
        logger.exception("TTS failed")
        raise HTTPException(status_code=500, detail=f"Lỗi tổng hợp: {type(e).__name__}: {e}")

    data = _wav_bytes(wav, sr)
    duration = len(wav) / sr if sr else 0.0
    rtf = (duration / elapsed) if elapsed > 0 else 0.0
    used_voice = voice or (manager.default_voice() or "default")
    status_msg = (
        f"✅ Hoàn tất! ({used_voice}, "
        f"{elapsed:.2f}s xử lý, {duration:.2f}s audio, {sr}Hz, {rtf:.2f}x realtime)"
    )
    # HTTP header values must be latin-1; Vietnamese/emoji status and the voice id
    # are percent-encoded (RFC 3986) so any client can decode them. Numeric headers
    # stay plain ASCII for easy reading.
    headers = {
        "X-Status": quote(status_msg),
        "X-Status-Encoding": "percent",
        "X-Duration-Sec": f"{duration:.3f}",
        "X-Elapsed-Sec": f"{elapsed:.3f}",
        "X-Realtime-Factor": f"{rtf:.3f}",
        "X-Sample-Rate": str(sr),
        "X-Voice": quote(used_voice),
        "X-Style": style,
        "Content-Disposition": 'inline; filename="vieneu_tts.wav"',
    }
    return Response(content=data, media_type="audio/wav", headers=headers)


@app.get("/api", include_in_schema=False)
def api_root() -> JSONResponse:
    """Tiện lợi: liệt kê nhanh các endpoint khi curl /api."""
    return JSONResponse(
        {
            "name": "VieNeu-TTS API",
            "version": "1.0.0",
            "docs": "/docs",
            "endpoints": [
                f"GET {API_PREFIX}/health",
                f"GET {API_PREFIX}/voices",
                f"GET {API_PREFIX}/styles",
                f"POST {API_PREFIX}/tts",
            ],
        }
    )


# ── Mount the Gradio UI at "/" (same port) ───────────────────────────────────
def _mount_ui() -> None:
    """Attach the existing Gradio Blocks at ``/`` so UI + API share one port.

    Best-effort: if Gradio import fails (e.g. a headless build) the API still
    runs; /health reports ui_mounted=false.
    """
    try:
        import gradio as gr
        from apps.gradio_main import demo

        gr.mount_gradio_app(app, demo, path="/")
        app.state.ui_mounted = True
        logger.info("🖥️  Gradio UI mounted at / (shared port).")
    except Exception as e:
        app.state.ui_mounted = False
        logger.warning("UI not mounted (%s). API still available under /api/v1.", e)


if os.getenv("VIENEU_API_MOUNT_UI", "1") == "1":
    _mount_ui()


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"))
    port = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
    logger.info("🚀 VieNeu-TTS API on http://%s:%d  (docs: /docs, api: %s)", host, port, API_PREFIX)
    # Behind a reverse proxy (nginx/Cloudflare) the app is reached over HTTPS but
    # uvicorn only sees plain HTTP from the proxy. Trust X-Forwarded-Proto/-For so
    # Gradio builds https:// asset URLs — otherwise the mounted UI tells the browser
    # to fetch audio over http:// on an https page and it's blocked as mixed content
    # (symptom: audio player stuck at 0:00, download fails, but /tts still works).
    uvicorn.run(app, host=host, port=port, proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
