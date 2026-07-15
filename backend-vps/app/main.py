"""FastAPI app for backend-vps: lifespan (DB + model-server probe), routers, docs.

Public API at ``/api/v1`` (no auth). The GPU model-server is one HTTP hop behind
the service layer; this process holds no model and stays torch-free.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .database import init_db
from .api import catalog, health, tts, voices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("Vieneu.VPS")

API_PREFIX = settings.API_PREFIX


class _ForceHttpsProto:
    """Rewrite x-forwarded-proto to https behind Cloudflare/nginx. Enable with
    VIENEU_FORCE_HTTPS=1 (needed if a proxy strips TLS and clients build URLs)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            scope = dict(scope)
            headers = [(n, v) for (n, v) in (scope.get("headers") or [])
                       if n.lower() not in (b"x-forwarded-proto", b"x-forwarded-scheme")]
            headers += [(b"x-forwarded-proto", b"https"), (b"x-forwarded-scheme", b"https")]
            scope["headers"] = headers
            scope["scheme"] = "https"
        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. DB up + tables.
    init_db()
    # 2. Recover jobs left mid-run by a previous process.
    from .services.job_service import recover_stale_jobs
    n = recover_stale_jobs()
    if n:
        logger.info("♻️  marked %d stale job(s) as error on startup", n)
    # 3. Probe the model-server (no model here — it lives on the GPU box).
    logger.info("🔗 model-server = %s", settings.MODEL_SERVER_URL)
    try:
        from .services.model_client import client
        h = client.health()
        logger.info("   model-server health: %s (%s/%s)",
                    h.get("status"), h.get("backend"), h.get("device"))
    except Exception as e:
        logger.warning("   model-server not reachable yet: %s", e)
    yield


app = FastAPI(
    title="VieNeu-TTS API",
    version="1.0.0",
    description=(
        "Backend TTS bất đồng bộ cho VieNeu-TTS v3 Turbo (48kHz). "
        "Job có uuid: tạo → poll % → hủy → tải WAV. CRUD giọng (voice cloning). "
        "Không cần token. Tổng hợp chạy trên GPU model-server."
    ),
    lifespan=lifespan,
    docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json",
)

if settings.FORCE_HTTPS:
    app.add_middleware(_ForceHttpsProto)

# Routers under /api/v1.
app.include_router(health.router, prefix=API_PREFIX)
app.include_router(catalog.router, prefix=API_PREFIX)
app.include_router(voices.router, prefix=API_PREFIX)
app.include_router(tts.router, prefix=API_PREFIX)


@app.get("/api", include_in_schema=False)
def api_root() -> JSONResponse:
    return JSONResponse({
        "name": "VieNeu-TTS API", "version": "1.0.0", "docs": "/docs",
        "endpoints": [
            f"GET  {API_PREFIX}/health",
            f"GET  {API_PREFIX}/voices", f"POST {API_PREFIX}/voices",
            f"GET  {API_PREFIX}/voices/{{id}}", f"DELETE {API_PREFIX}/voices/{{id}}",
            f"GET  {API_PREFIX}/styles", f"GET  {API_PREFIX}/modes",
            f"POST {API_PREFIX}/tts", f"GET  {API_PREFIX}/tts/{{id}}",
            f"DELETE {API_PREFIX}/tts/{{id}}", f"GET  {API_PREFIX}/tts/{{id}}/download",
        ],
    })


# Demo customer UI at "/" (static files), if present.
def _mount_demo() -> None:
    demo_dir = Path(__file__).resolve().parents[1] / "demo"
    if settings.MOUNT_DEMO and demo_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(demo_dir), html=True), name="demo")
        logger.info("🖥️  demo UI mounted at / (%s)", demo_dir)


_mount_demo()


def main() -> None:
    import uvicorn
    logger.info("🚀 VieNeu-TTS API on http://%s:%d (docs: /docs, api: %s)",
                settings.HOST, settings.PORT, API_PREFIX)
    uvicorn.run(app, host=settings.HOST, port=settings.PORT,
                proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
