"""main — entrypoint kích hoạt ứng dụng (tầng ngoài cùng của server/).

Chạy ``main.py`` bật MỘT server trên MỘT cổng (PORT=7862):
  • Giao diện tại  ``/``          — phục vụ repo-root/ui/index.html (cùng origin).
  • API tại        ``/api/v1``    — + /docs Swagger + /files.

Cùng cổng vì môi trường deploy (CT3600) chỉ mở đúng 1 cổng 7862. Giao diện gọi
API bằng đường DẪN TƯƠNG ĐỐI (cùng origin) nên không cần CORS và chạy được sau
mọi proxy/domain. Tắt giao diện: ``VIENEU_SERVE_UI=0``.

Chạy:
    uv run python server/main.py
    # hoặc: uv run uvicorn server.main:app --host 0.0.0.0 --port 7862
"""
from __future__ import annotations

import logging
import sys
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
    # DB (PostgreSQL) — lưu giọng custom + jobs của user. Tạo bảng trước.
    from src.db import init_db
    from src.repositories import jobs_repo
    init_db()
    # Job còn queued/running từ lần chạy trước = mồ côi (RAM đã mất) → đánh dấu error.
    n = jobs_repo.mark_interrupted()
    if n:
        logger.warning("⚠️ %d job mồ côi từ lần chạy trước → đánh dấu error.", n)
    if settings.MODEL_EAGER_LOAD:
        try:
            engine.load()   # preset nạp vào RAM (dùng chung, KHÔNG vào DB)
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

# CORS — giao diện giờ CÙNG origin (phục vụ tại "/"), không cần CORS. Giữ lại
# cho trường hợp gọi API từ origin khác (đặt VIENEU_API_BASE / client ngoài).
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


# ── Giao diện UI — phục vụ tại "/" NGAY TRÊN app API (cùng cổng 7862) ─────────
# BE không dùng "/" (chỉ /api, /api/v1/*, /files, /docs) nên giao diện ở gốc.
# HTML gọi API bằng đường dẫn tương đối (window.__API_BASE__ để rỗng = cùng
# origin) → không cần CORS, chạy sau mọi proxy/domain của CT3600.
if settings.SERVE_UI:
    _UI_INDEX = Path(settings.UI_DIR) / "index.html"

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui_index() -> HTMLResponse:
        if not _UI_INDEX.is_file():
            return HTMLResponse(f"<h1>Không tìm thấy giao diện</h1><p>{_UI_INDEX}</p>",
                                status_code=404)
        html = _UI_INDEX.read_text(encoding="utf-8")
        # API_BASE_URL mặc định rỗng → giao diện gọi API tương đối, cùng origin.
        inject = f'<script>window.__API_BASE__="{settings.API_BASE_URL}";</script>'
        html = html.replace("</head>", f"{inject}\n</head>", 1)
        return HTMLResponse(html)


def main() -> None:
    import uvicorn

    ui_note = f"UI: /  ·  " if settings.SERVE_UI else ""
    logger.info("🚀 VieNeu-TTS on http://%s:%d (%sdocs: /docs, api: %s)",
                settings.HOST, settings.PORT, ui_note, API_PREFIX)
    uvicorn.run(app, host=settings.HOST, port=settings.PORT,
                proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
