"""Health + readiness (mode-aware)."""
from __future__ import annotations

from fastapi import APIRouter

from ..config import settings
from ..db import engine
from ..services.jobs import manager
from ..schemas import HealthResponse
from ..services.tts_service import service
from ..services import catalog_provider as catalog

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Sức khỏe + kiến trúc")
def health() -> HealthResponse:
    db_ok = True
    try:
        with engine.connect() as c:
            c.exec_driver_sql("SELECT 1")
    except Exception:
        db_ok = False

    if settings.MODE == "remote":
        # Reflect the model-server's health.
        backend = device = sr = None
        loaded = False
        num_voices = 0
        try:
            from ..services.model_client import client
            h = client.health()
            loaded = bool(h.get("model_loaded"))
            backend, device, sr = h.get("backend"), h.get("device"), h.get("sample_rate")
            num_voices = h.get("num_voices", 0)
        except Exception:
            loaded = False
        status = "ok" if (loaded and db_ok) else "loading"
        return HealthResponse(status=status, model_loaded=loaded, backend=backend,
                              device=device, backbone_repo=settings.BACKBONE_REPO, sample_rate=sr,
                              num_voices=num_voices, db_connected=db_ok, active_jobs=manager.active_count())

    loaded = service.loaded
    status = "ok" if (loaded and db_ok) else ("error" if service.load_error else "loading")
    return HealthResponse(
        status=status, model_loaded=loaded, backend=service.backend, device=service.device,
        backbone_repo=settings.BACKBONE_REPO, sample_rate=service.sample_rate,
        num_voices=len(service.list_voices()), db_connected=db_ok, active_jobs=manager.active_count(),
    )
