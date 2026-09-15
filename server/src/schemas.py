"""Pydantic request/response models — 1 file schema tập trung.

Toàn bộ shape của API /api/v1 nằm ở đây (drive Swagger/OpenAPI). Controller và
services import từ module này để giữ 1 nguồn sự thật duy nhất.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import List, Literal, Optional
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

# ── Hằng dùng chung ───────────────────────────────────────────────────────────
STYLE_CHOICES = {
    "tu_nhien": "Tự nhiên (conversational)",
    "tin_tuc": "Tin tức (news)",
    "doc_truyen": "Kể chuyện (storytelling)",
}
DEFAULT_STYLE = "tu_nhien"

# Mode xử lý: cpu (in-process, mặc định) | gpu (Vast.ai on-demand).
MODE_CPU = "cpu"
MODE_GPU = "gpu"
MODE_AUTO = "auto"          # BE tự chọn theo độ dài context
MODE_CHOICES = (MODE_CPU, MODE_GPU, MODE_AUTO)
MAX_DELIVERY_BYTES = 256 * 1024 * 1024

_DELIVERY_HEADER_NAMES = {"Content-Type", "Content-MD5", "If-None-Match"}
_CONTENT_MD5_PATTERN = r"^[A-Za-z0-9+/]{22}==$"


# ── TTS jobs ──────────────────────────────────────────────────────────────────
class DeliveryCapability(BaseModel):
    generation_id: UUID
    capability_token: str = Field(min_length=43, max_length=128)


class DeliveryGrant(BaseModel):
    method: Literal["PUT"]
    url: str = Field(min_length=1, max_length=4096)
    headers: dict[Literal["Content-Type", "Content-MD5", "If-None-Match"], str]
    expires_at: datetime

    @field_validator("url")
    @classmethod
    def validate_raw_url(cls, value: str) -> str:
        if any(ord(char) <= 32 for char in value):
            raise ValueError("delivery URL contains whitespace or control characters")
        try:
            parsed = urlsplit(value)
            hostname = parsed.hostname
            _port = parsed.port
        except ValueError as exc:
            raise ValueError("delivery URL is invalid") from exc
        if parsed.scheme not in {"http", "https"} or not hostname:
            raise ValueError("delivery URL must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.fragment:
            raise ValueError("delivery URL must not contain userinfo or a fragment")
        return value

    @field_validator("headers")
    @classmethod
    def validate_headers(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) != _DELIVERY_HEADER_NAMES:
            raise ValueError("delivery grant headers must match the upload allowlist")
        if value["Content-Type"] != "audio/wav":
            raise ValueError("Content-Type must be audio/wav")
        if value["If-None-Match"] != "*":
            raise ValueError("If-None-Match must be *")
        if re.fullmatch(_CONTENT_MD5_PATTERN, value["Content-MD5"]) is None:
            raise ValueError("Content-MD5 must be a base64-encoded MD5 digest")
        return value

    @field_validator("expires_at")
    @classmethod
    def validate_expiry_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("expires_at must include a timezone")
        return value


class DeliveryComplete(BaseModel):
    upstream_job_id: UUID
    status: Literal["ready", "error"]
    audio_size_bytes: Optional[int] = Field(default=None, gt=0, le=MAX_DELIVERY_BYTES)
    object_etag: Optional[str] = Field(default=None, max_length=256)
    duration_sec: Optional[float] = Field(default=None, ge=0)
    elapsed_sec: Optional[float] = Field(default=None, ge=0)
    sample_rate: Optional[int] = Field(default=None, gt=0)
    error: Optional[str] = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_terminal_state(self) -> DeliveryComplete:
        if self.status == "ready":
            required = (self.audio_size_bytes, self.duration_sec, self.sample_rate)
            if any(value is None for value in required) or self.error is not None:
                raise ValueError("ready delivery requires audio metadata and no error")
            return self

        media_fields = (
            self.audio_size_bytes,
            self.object_etag,
            self.duration_sec,
            self.sample_rate,
        )
        if self.error is None or any(value is not None for value in media_fields):
            raise ValueError("error delivery requires an error and null audio metadata")
        return self


class TTSCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000,
                      description="Văn bản cần đọc. Có thể chèn [cười]/[thở dài]/[hắng giọng].",
                      examples=["Xin chào, đây là VieNeu-TTS."])
    voice: Optional[str] = Field(default=None,
                                 description="ID giọng (xem GET /voices). Bỏ trống = mặc định.",
                                 examples=["Mai Anh"])
    style: str = Field(default=DEFAULT_STYLE, description="tu_nhien | tin_tuc | doc_truyen.")
    temperature: float = Field(default=0.8, ge=0.1, le=1.5)
    max_chars: int = Field(default=256, ge=32, le=512, description="Số ký tự tối đa mỗi chunk.")
    mode: str = Field(default=MODE_AUTO,
                      description="cpu | gpu | auto. gpu chỉ hợp lệ khi context ≥ ngưỡng số từ; "
                                  "auto = BE tự chọn.")
    name_audio: Optional[str] = Field(
        default=None, max_length=50,
        description="Tên hiển thị của file audio. Bỏ trống = {job_id}.wav.")
    delivery: Optional[DeliveryCapability] = None


class JobCreated(BaseModel):
    id: str = Field(..., description="Job id (uuid). Dùng để poll / download / cancel.")
    status: str
    mode: str = Field(..., description="Mode thực tế đã chọn: cpu | gpu.")
    name_audio: str
    audio_key: Optional[str] = None
    audio_size_bytes: Optional[int] = None
    delivery_kind: str = "vieneu"
    external_generation_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    poll_url: str
    download_url: Optional[str] = None


class JobStatus(BaseModel):
    id: str
    status: str = Field(..., description="queued | running | done | cancelled | error")
    mode: str = Field(default=MODE_CPU, description="cpu | gpu")
    progress: float = Field(..., description="Tiến độ %, 0–100.")
    done_chunks: int
    total_chunks: int
    voice: Optional[str] = None
    style: str
    name_audio: Optional[str] = None
    audio_key: Optional[str] = None
    audio_size_bytes: Optional[int] = None
    duration_sec: Optional[float] = None
    elapsed_sec: Optional[float] = None
    sample_rate: Optional[int] = None
    error: Optional[str] = None
    delivery_kind: str = "vieneu"
    external_generation_id: Optional[str] = None
    delivery_unconfirmed: bool = False
    # ── Chỉ có ở GPU mode (truy vết máy Vast.ai + tiền) ──────────────────────
    instance_id: Optional[int] = Field(default=None, description="(GPU) id máy Vast.ai đang thuê.")
    dph: Optional[float] = Field(default=None, description="(GPU) giá $/giờ của máy đang thuê.")
    est_cost: Optional[float] = Field(default=None, description="(GPU) tiền ước tính đã tốn ($).")
    download_url: Optional[str] = Field(default=None, description="Có khi status=done.")
    created_at: datetime
    updated_at: datetime


class JobInfo(BaseModel):
    id: str
    user_ref: str
    text: str
    voice: Optional[str] = None
    style: str
    temperature: float
    max_chars: int
    mode: str
    status: str
    progress: float
    done_chunks: int
    total_chunks: int
    name_audio: Optional[str] = None
    audio_key: Optional[str] = None
    audio_size_bytes: Optional[int] = None
    duration_sec: Optional[float] = None
    elapsed_sec: Optional[float] = None
    sample_rate: Optional[int] = None
    instance_id: Optional[int] = None
    error: Optional[str] = None
    download_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class JobsSummary(BaseModel):
    total_jobs: int = 0
    queued_jobs: int = 0
    running_jobs: int = 0
    done_jobs: int = 0
    cancelled_jobs: int = 0
    error_jobs: int = 0
    cpu_jobs: int = 0
    gpu_jobs: int = 0
    total_duration_sec: float = 0
    total_elapsed_sec: float = 0
    total_audio_size_bytes: int = 0


class JobsResponse(BaseModel):
    summary: JobsSummary
    jobs: List[JobInfo]


class JobUpdate(BaseModel):
    name_audio: str = Field(..., min_length=1, max_length=50,
                            description="Tên hiển thị mới của file audio.")


class JobDeleteResponse(BaseModel):
    id: str
    deleted_job: bool
    deleted_audio: bool
    audio_key: Optional[str] = None


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
    active_jobs: int = 0
    gpu_min_words: int = Field(default=1000, description="Ngưỡng số từ tối thiểu để cho phép GPU mode.")


class ErrorResponse(BaseModel):
    detail: str
