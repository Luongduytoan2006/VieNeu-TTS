"""Catalog: reading styles + synthesis modes."""
from __future__ import annotations

from fastapi import APIRouter

from ..schemas import DEFAULT_STYLE, STYLE_CHOICES, ModeInfo, ModesResponse, StyleInfo, StylesResponse
from ..services.tts_service import service

router = APIRouter(tags=["catalog"])


@router.get("/styles", response_model=StylesResponse, summary="Phong cách đọc")
def styles() -> StylesResponse:
    items = [StyleInfo(id=sid, label=label, is_default=(sid == DEFAULT_STYLE))
             for sid, label in STYLE_CHOICES.items()]
    return StylesResponse(count=len(items), default_style=DEFAULT_STYLE, styles=items)


@router.get("/modes", response_model=ModesResponse, summary="Chế độ tổng hợp")
def modes() -> ModesResponse:
    """Backend đang chạy (v3 Turbo). Mô tả các mode SDK hỗ trợ; mode active tùy máy
    (GPU→PyTorch, CPU→ONNX)."""
    active = service.backend or "unknown"
    items = [
        ModeInfo(id="v3turbo", label="VieNeu-TTS v3 Turbo (48kHz)",
                 description="Mặc định. GPU dùng PyTorch, CPU dùng ONNX. Voice cloning, tag cảm xúc.",
                 available=True),
        ModeInfo(id="pytorch", label="PyTorch (GPU)",
                 description="Engine PyTorch cho GPU CUDA.", available=(service.device == "cuda")),
        ModeInfo(id="onnx", label="ONNX (CPU)",
                 description="Engine ONNX torch-free cho CPU.", available=(service.device == "cpu")),
    ]
    return ModesResponse(count=len(items), active_mode=active, modes=items)
