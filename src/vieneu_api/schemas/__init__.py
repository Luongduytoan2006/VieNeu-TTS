"""Pydantic request/response models — drive the OpenAPI/Swagger docs."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

STYLE_CHOICES = {
    "tu_nhien": "Tự nhiên (conversational)",
    "tin_tuc": "Tin tức (news)",
    "doc_truyen": "Kể chuyện (storytelling)",
}
DEFAULT_STYLE = "tu_nhien"


# ── TTS jobs ──────────────────────────────────────────────────────────────────
class TTSCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000,
                      description="Văn bản cần đọc. Có thể chèn [cười]/[thở dài]/[hắng giọng].",
                      examples=["Xin chào, đây là VieNeu-TTS."])
    voice: Optional[str] = Field(default=None, description="ID giọng (xem GET /voices). Bỏ trống = mặc định.",
                                 examples=["Phạm Tuyên"])
    style: str = Field(default=DEFAULT_STYLE, description="tu_nhien | tin_tuc | doc_truyen.")
    temperature: float = Field(default=0.8, ge=0.1, le=1.5)
    max_chars: int = Field(default=256, ge=32, le=512, description="Số ký tự tối đa mỗi chunk.")


class JobCreated(BaseModel):
    id: str = Field(..., description="Job id (uuid). Dùng để poll / download / cancel.")
    status: str
    poll_url: str
    download_url: str


class JobStatus(BaseModel):
    id: str
    status: str = Field(..., description="queued | running | done | cancelled | error")
    progress: float = Field(..., description="Tiến độ %, 0–100.")
    done_chunks: int
    total_chunks: int
    voice: Optional[str] = None
    style: str
    duration_sec: Optional[float] = None
    elapsed_sec: Optional[float] = None
    sample_rate: Optional[int] = None
    error: Optional[str] = None
    download_url: Optional[str] = Field(default=None, description="Có khi status=done.")
    created_at: datetime
    updated_at: datetime


# ── Voices ────────────────────────────────────────────────────────────────────
class VoiceInfo(BaseModel):
    id: str
    label: str
    gender: Optional[str] = None
    style: str = DEFAULT_STYLE
    source: str = "preset"          # preset | custom
    is_default: bool = False


class VoicesResponse(BaseModel):
    count: int
    default_voice: Optional[str] = None
    voices: List[VoiceInfo]


# ── Catalog ───────────────────────────────────────────────────────────────────
class StyleInfo(BaseModel):
    id: str
    label: str
    is_default: bool = False


class StylesResponse(BaseModel):
    count: int
    default_style: str
    styles: List[StyleInfo]


class ModeInfo(BaseModel):
    id: str
    label: str
    description: str
    available: bool


class ModesResponse(BaseModel):
    count: int
    active_mode: str
    modes: List[ModeInfo]


# ── Health ────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    backend: Optional[str] = None
    device: Optional[str] = None
    backbone_repo: Optional[str] = None
    sample_rate: Optional[int] = None
    num_voices: int = 0
    db_connected: bool = False
    active_jobs: int = 0


class ErrorResponse(BaseModel):
    detail: str
