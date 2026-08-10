"""System and catalog routes — health, styles, modes."""
from __future__ import annotations

from fastapi import APIRouter

from ..config import settings
from ..schemas import HealthResponse, ModesResponse, StylesResponse
from ..services import catalog as catalog_svc

router = APIRouter(prefix=settings.API_PREFIX)


@router.get("/health", response_model=HealthResponse, tags=["system"],
            summary="GET /api/v1/health — Sức khỏe + kiến trúc")
def health() -> HealthResponse:
    return catalog_svc.health()


@router.get("/styles", response_model=StylesResponse, tags=["catalog"],
            summary="GET /api/v1/styles — Phong cách đọc")
def styles() -> StylesResponse:
    return catalog_svc.list_styles()


@router.get("/modes", response_model=ModesResponse, tags=["catalog"],
            summary="GET /api/v1/modes — Chế độ xử lý (cpu | gpu)")
def modes() -> ModesResponse:
    return catalog_svc.list_modes()
