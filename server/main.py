"""main — entrypoint kích hoạt ứng dụng (tầng ngoài cùng của server/).

Chạy ``main.py`` bật CÙNG LÚC 2 server trên 2 cổng khác nhau:
  • API  (FastAPI, cổng PORT=7862)      — /api/v1 + /docs Swagger + /files.
  • Giao diện (static, cổng DEMO_PORT=7870) — phục vụ repo-root/demo/index.html,
    file này gọi ngược lại API qua CORS.

Chạy:
    uv run python server/main.py
    # hoặc chỉ API: uv run uvicorn server.main:app --host 0.0.0.0 --port 7862
"""
from __future__ import annotations

import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

# Bootstrap: cho phép import ``src`` / ``vieneu`` / ``vieneu_utils`` dù chạy từ
# repo root (uvicorn server.main:app) hay từ trong server/ (python main.py).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from src.config import settings
from src.controller import router as api_router
from src.engine import engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("Vieneu.Server")

API_PREFIX = settings.API_PREFIX


class _ForceHttpsProto:
    """Rewrite x-forwarded-proto → https sau Cloudflare/nginx. Bật VIENEU_FORCE_HTTPS=1."""

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
    logger.info("🔧 CPU mode in-process; GPU mode qua Vast.ai on-demand (ngưỡng %d từ).",
                settings.GPU_MIN_WORDS)
    if settings.MODEL_EAGER_LOAD:
        try:
            engine.load()
        except Exception:
            logger.error("Nạp model ở startup thất bại; /api/v1/health sẽ báo lỗi.")
    yield


app = FastAPI(
    title="VieNeu-TTS API",
    version="1.0.0",
    description=(
        "Backend TTS bất đồng bộ cho VieNeu-TTS v3 Turbo (48kHz). "
        "Job có uuid: tạo → poll % → hủy → tải WAV. CRUD giọng (voice cloning). "
        "2 mode: **CPU** (in-process, mặc định) và **GPU** (Vast.ai on-demand, context dài). "
        "Không cần token."
    ),
    lifespan=lifespan,
    docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json",
)

if settings.FORCE_HTTPS:
    app.add_middleware(_ForceHttpsProto)

# CORS — giao diện chạy ở cổng khác nên trình duyệt cần API cho phép cross-origin.
_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


# ── /files/{key} — phục vụ audio khi storage=local ────────────────────────────
@app.get("/files/{key:path}", include_in_schema=False)
def get_file(key: str):
    from src.storage import get_storage
    st = get_storage()
    if not st.exists(key):
        raise HTTPException(404, "Không tìm thấy file.")
    return Response(content=st.open(key), media_type="audio/wav",
                    headers={"Content-Disposition": f'inline; filename="{key.split("/")[-1]}"'})


@app.get("/api", include_in_schema=False)
def api_root() -> JSONResponse:
    return JSONResponse({
        "name": "VieNeu-TTS API", "version": "1.0.0", "docs": "/docs",
        "endpoints": [
            f"GET  {API_PREFIX}/health",
            f"GET  {API_PREFIX}/styles", f"GET  {API_PREFIX}/modes",
            f"GET  {API_PREFIX}/voices", f"POST {API_PREFIX}/voices",
            f"GET  {API_PREFIX}/voices/{{id}}", f"DELETE {API_PREFIX}/voices/{{id}}",
            f"POST {API_PREFIX}/tts", f"GET  {API_PREFIX}/tts/{{id}}",
            f"DELETE {API_PREFIX}/tts/{{id}}", f"GET  {API_PREFIX}/tts/{{id}}/download",
        ],
    })


# ── Giao diện demo — app RIÊNG, chạy CỔNG RIÊNG (tách ngoài server/) ──────────
def build_demo_app() -> FastAPI:
    """Dựng app tĩnh phục vụ repo-root/demo/index.html.

    Trang này gọi ngược lại API. URL API được nhúng vào HTML lúc phục vụ (biến
    ``window.__API_BASE__``) để giao diện biết cổng API mà không hardcode.
    """
    demo = FastAPI(title="VieNeu-TTS Demo UI", docs_url=None, redoc_url=None,
                   openapi_url=None)
    demo_dir = Path(settings.DEMO_DIR)
    index = demo_dir / "index.html"

    @demo.get("/", response_class=HTMLResponse)
    def _index() -> HTMLResponse:
        if not index.is_file():
            return HTMLResponse(f"<h1>Không tìm thấy giao diện</h1><p>{index}</p>", status_code=404)
        html = index.read_text(encoding="utf-8")
        # Nhúng API base để HTML gọi đúng cổng API (khác cổng giao diện).
        inject = f'<script>window.__API_BASE__="{settings.API_BASE_URL}";</script>'
        html = html.replace("</head>", f"{inject}\n</head>", 1)
        return HTMLResponse(html)

    return demo


def _run_demo() -> None:
    import uvicorn
    demo_app = build_demo_app()
    logger.info("🖥️  Giao diện demo on http://%s:%d → API %s",
                settings.HOST, settings.DEMO_PORT, settings.API_BASE_URL)
    uvicorn.run(demo_app, host=settings.HOST, port=settings.DEMO_PORT,
                log_level="warning")


def main() -> None:
    import uvicorn

    # Giao diện chạy ở thread nền (cổng riêng); API giữ luồng chính.
    if settings.SERVE_DEMO:
        threading.Thread(target=_run_demo, daemon=True, name="demo-ui").start()

    logger.info("🚀 VieNeu-TTS API on http://%s:%d (docs: /docs, api: %s)",
                settings.HOST, settings.PORT, API_PREFIX)
    uvicorn.run(app, host=settings.HOST, port=settings.PORT,
                proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
