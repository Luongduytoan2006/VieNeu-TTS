"""Catalog: reading styles + synthesis modes."""
from __future__ import annotations

from fastapi import APIRouter

from ..schemas import ModesResponse, StylesResponse
from ..services import catalog_service

router = APIRouter(tags=["catalog"])


@router.get("/styles", response_model=StylesResponse, summary="Phong cách đọc")
def styles() -> StylesResponse:
    return catalog_service.list_styles()


@router.get("/modes", response_model=ModesResponse, summary="Chế độ tổng hợp")
def modes() -> ModesResponse:
    return catalog_service.list_modes()
