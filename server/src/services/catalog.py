"""catalog — dữ liệu tra cứu: styles, modes, health. Đọc từ engine in-process."""
from __future__ import annotations

from ..config import settings
from ..engine import engine
from ..repositories import voices_repo as repo
from ..schemas import (
    DEFAULT_STYLE, STYLE_CHOICES, HealthResponse, ModeInfo, ModesResponse,
    StyleInfo, StylesResponse,
)
from .jobs import manager


def list_styles() -> StylesResponse:
    items = [StyleInfo(id=sid, label=label, is_default=(sid == DEFAULT_STYLE))
             for sid, label in STYLE_CHOICES.items()]
    return StylesResponse(count=len(items), default_style=DEFAULT_STYLE, styles=items)


def list_modes() -> ModesResponse:
    """2 chế độ xử lý: cpu (in-process, sẵn sàng) | gpu (Vast.ai on-demand)."""
    device = engine.device
    cpu_ready = engine.loaded
    items = [
        ModeInfo(id="cpu", label="CPU (in-process)",
                 description=f"Model nạp sẵn trên server này ({device or 'cpu'}). "
                             f"Mặc định, dùng cho mọi văn bản.",
                 available=cpu_ready),
        ModeInfo(id="gpu", label="GPU (Vast.ai on-demand)",
                 description=f"Tạo máy GPU thuê theo giờ khi context ≥ {settings.GPU_MIN_WORDS} "
                             f"từ; chạy xong hủy máy.",
                 available=bool(settings.VAST_AI_API_KEY)),
    ]
    active = "cpu" if cpu_ready else "loading"
    return ModesResponse(count=len(items), active_mode=active, modes=items)


def health() -> HealthResponse:
    loaded = engine.loaded
    status = "ok" if loaded else ("error" if engine.load_error else "loading")
    return HealthResponse(
        status=status, model_loaded=loaded,
        backend=engine.backend, device=engine.device,
        backbone_repo=settings.BACKBONE_REPO,
        sample_rate=engine.sample_rate if loaded else None,
        num_voices=repo.count(),
        active_jobs=manager.active_count(),
        gpu_min_words=settings.GPU_MIN_WORDS,
    )


def ready() -> bool:
    return engine.loaded
