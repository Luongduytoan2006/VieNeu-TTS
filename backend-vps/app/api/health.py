"""Health + readiness — reflects the model-server's health plus our DB."""
from __future__ import annotations

from fastapi import APIRouter

from ..config import settings
from ..database import engine
from ..schemas import HealthResponse
from ..services import catalog_service
from ..services.job_service import manager

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Sức khỏe + kiến trúc")
def health() -> HealthResponse:
    db_ok = True
    try:
        with engine.connect() as c:
            c.exec_driver_sql("SELECT 1")
    except Exception:
        db_ok = False

    h = catalog_service.model_health()
    loaded = bool(h.get("model_loaded"))
    status = "ok" if (loaded and db_ok) else "loading"
    return HealthResponse(
        status=status, model_loaded=loaded,
        backend=h.get("backend"), device=h.get("device"),
        backbone_repo=settings.BACKBONE_REPO, sample_rate=h.get("sample_rate"),
        num_voices=h.get("num_voices", 0), db_connected=db_ok,
        active_jobs=manager.active_count(),
    )
