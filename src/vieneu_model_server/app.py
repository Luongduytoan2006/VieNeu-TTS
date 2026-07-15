"""Model-server HTTP API (the GPU half). Auth via a shared bearer key."""
from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from .engine import DEFAULT_STYLE, STYLE_CHOICES, engine
from .jobs import STORAGE, manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("Vieneu.ModelServer")

PREFIX = "/model/v1"
API_KEY = os.getenv("MODEL_API_KEY", "dev-secret-key")
UPLOAD_DIR = Path(os.getenv("MODEL_UPLOAD_DIR", str(Path(__file__).resolve().parents[2] / ".ask" / "storage" / "uploads")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def require_key(authorization: Optional[str] = Header(default=None)) -> None:
    """Shared-secret bearer auth between BE and model-server."""
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(401, "Sai hoặc thiếu API key.")


# ── Schemas ──────────────────────────────────────────────────────────────────
class SynthReq(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)
    voice: Optional[str] = None
    style: str = DEFAULT_STYLE
    temperature: float = Field(default=0.8, ge=0.1, le=1.5)
    max_chars: int = Field(default=256, ge=32, le=512)


class JobResp(BaseModel):
    id: str
    status: str
    progress: float
    done_chunks: int
    total_chunks: int
    audio_url: Optional[str] = None
    audio_key: Optional[str] = None
    duration_sec: Optional[float] = None
    elapsed_sec: Optional[float] = None
    sample_rate: Optional[int] = None
    error: Optional[str] = None


def _job_resp(j) -> JobResp:
    return JobResp(id=j.id, status=j.status, progress=j.progress, done_chunks=j.done_chunks,
                   total_chunks=j.total_chunks, audio_url=j.audio_url, audio_key=j.audio_key,
                   duration_sec=j.duration_sec, elapsed_sec=j.elapsed_sec,
                   sample_rate=j.sample_rate, error=j.error)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("MODEL_EAGER_LOAD", "1") == "1":
        try:
            engine.load()
        except Exception:
            logger.error("model load failed at startup; /health will report it")
    yield


app = FastAPI(title="VieNeu-TTS Model Server", version="1.0.0",
              description="GPU-half: sinh audio, lưu storage, trả URL. Auth bằng bearer key.",
              lifespan=lifespan)


@app.get(f"{PREFIX}/health")
def health():
    return {
        "status": "ok" if engine.loaded else ("error" if engine.load_error else "loading"),
        "model_loaded": engine.loaded, "backend": engine.backend, "device": engine.device,
        "sample_rate": engine.sample_rate if engine.loaded else None,
        "num_voices": len(engine.list_voices()), "active_jobs": manager.active_count(),
        "storage": type(STORAGE).__name__,
    }


@app.get(f"{PREFIX}/voices", dependencies=[Depends(require_key)])
def voices():
    dv = engine.default_voice()
    out = []
    for label, vid in engine.list_voices():
        m = engine.voice_meta(vid)
        out.append({"id": vid, "label": label, "gender": m.get("gender") or None,
                    "style": m.get("style", DEFAULT_STYLE), "is_default": vid == dv})
    return {"count": len(out), "default_voice": dv, "voices": out}


@app.post(f"{PREFIX}/voices", dependencies=[Depends(require_key)], status_code=201)
async def enroll_voice(name: str = Form(...), audio: UploadFile = File(...),
                       description: str = Form(""), gender: str = Form(""),
                       style: str = Form(DEFAULT_STYLE), denoise: bool = Form(True)):
    if not engine.loaded:
        raise HTTPException(503, "Model chưa sẵn sàng.")
    if engine.has_voice(name):
        raise HTTPException(409, f"Voice '{name}' đã tồn tại.")
    ext = ("." + audio.filename.rsplit(".", 1)[-1].lower()) if audio.filename and "." in audio.filename else ".wav"
    dest = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
    data = await audio.read()
    if not data:
        raise HTTPException(422, "File rỗng.")
    dest.write_bytes(data)
    try:
        engine.enroll(name, str(dest), description=description, gender=gender, style=style, denoise=denoise)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(500, f"Lỗi phân tích giọng: {type(e).__name__}: {e}")
    return {"id": name, "description": description, "gender": gender or None, "style": style}


@app.delete(f"{PREFIX}/voices/{{voice_id}}", dependencies=[Depends(require_key)], status_code=204)
def delete_voice(voice_id: str):
    if not engine.has_voice(voice_id):
        raise HTTPException(404, f"Voice '{voice_id}' không tồn tại.")
    engine.remove(voice_id)
    return None


@app.post(f"{PREFIX}/jobs", dependencies=[Depends(require_key)], response_model=JobResp, status_code=202)
def create_job(req: SynthReq):
    if not engine.loaded:
        raise HTTPException(503, "Model chưa sẵn sàng.")
    if req.voice and not engine.has_voice(req.voice):
        raise HTTPException(422, f"Voice '{req.voice}' không tồn tại.")
    style = req.style if req.style in STYLE_CHOICES else DEFAULT_STYLE
    job = manager.create(req.text, req.voice, style, req.temperature, req.max_chars)
    return _job_resp(job)


@app.get(f"{PREFIX}/jobs/{{job_id}}", dependencies=[Depends(require_key)], response_model=JobResp)
def get_job(job_id: str):
    j = manager.get(job_id)
    if j is None:
        raise HTTPException(404, f"Job '{job_id}' không tồn tại.")
    return _job_resp(j)


@app.delete(f"{PREFIX}/jobs/{{job_id}}", dependencies=[Depends(require_key)], response_model=JobResp)
def cancel_job(job_id: str):
    j = manager.get(job_id)
    if j is None:
        raise HTTPException(404, f"Job '{job_id}' không tồn tại.")
    manager.cancel(job_id)
    return _job_resp(j)


# Local storage file server (used when STORAGE backend = local). For R2/S3 the
# client would hit the bucket URL directly instead.
@app.get("/files/{key:path}")
def get_file(key: str):
    if not STORAGE.exists(key):
        raise HTTPException(404, "Không tìm thấy file.")
    return Response(content=STORAGE.open(key), media_type="audio/wav",
                    headers={"Content-Disposition": f'inline; filename="{key.split("/")[-1]}"'})


@app.get("/", include_in_schema=False)
def root():
    return JSONResponse({"name": "VieNeu-TTS Model Server", "docs": "/docs", "prefix": PREFIX})


def main() -> None:
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("MODEL_PORT", "9000")))
    logger.info("🚀 Model-server on http://%s:%d (prefix %s)", host, port, PREFIX)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
