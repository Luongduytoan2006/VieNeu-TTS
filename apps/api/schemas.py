"""Pydantic request/response models — these drive the OpenAPI/Swagger docs.

Every field carries a description + example so ``/docs`` is self-explanatory to a
caller who has never seen the code (e.g. testing from Postman on a phone).
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ── Reading styles (fixed set the v3 Turbo model was trained with) ──────────────
# label = what a human sees, id = what the model/infer() expects.
STYLE_CHOICES = {
    "tu_nhien": "Tự nhiên (conversational)",
    "tin_tuc": "Tin tức (news)",
    "doc_truyen": "Kể chuyện (storytelling)",
}
DEFAULT_STYLE = "tu_nhien"


class TTSRequest(BaseModel):
    """Body for ``POST /api/v1/tts`` — synthesize one piece of text."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Văn bản cần đọc (tiếng Việt/Anh). Có thể chèn tag cảm xúc "
        "[cười] / [thở dài] / [hắng giọng].",
        examples=["Xin chào, đây là giọng nói được tạo bởi VieNeu-TTS."],
    )
    voice: Optional[str] = Field(
        default=None,
        description="ID giọng preset (xem GET /api/v1/voices). Bỏ trống = dùng "
        "giọng mặc định.",
        examples=["Phạm Tuyên"],
    )
    style: str = Field(
        default=DEFAULT_STYLE,
        description=f"Phong cách đọc. Một trong: {', '.join(STYLE_CHOICES)}.",
        examples=[DEFAULT_STYLE],
    )
    temperature: float = Field(
        default=0.8,
        ge=0.1,
        le=1.5,
        description="Độ ngẫu nhiên khi sinh (0.1–1.5). Cao hơn = biến thiên hơn.",
    )
    max_chars: int = Field(
        default=256,
        ge=32,
        le=512,
        description="Số ký tự tối đa mỗi đoạn trước khi cắt chunk.",
    )
    apply_watermark: bool = Field(
        default=True,
        description="Đóng dấu bản quyền ẩn (Perth) vào audio.",
    )

    def validated_style(self) -> str:
        """Return a style id the model accepts, falling back to the default."""
        return self.style if self.style in STYLE_CHOICES else DEFAULT_STYLE


class VoiceInfo(BaseModel):
    """One preset voice."""

    id: str = Field(..., description="ID dùng khi gọi POST /api/v1/tts.", examples=["Phạm Tuyên"])
    label: str = Field(..., description="Tên hiển thị / mô tả.", examples=["Phạm Tuyên — nam miền Bắc"])
    gender: Optional[str] = Field(default=None, description="Giới tính giọng nếu có.")
    is_default: bool = Field(default=False, description="Có phải giọng mặc định không.")


class VoicesResponse(BaseModel):
    count: int = Field(..., description="Số giọng khả dụng.")
    default_voice: Optional[str] = Field(default=None, description="ID giọng mặc định.")
    voices: List[VoiceInfo]


class StyleInfo(BaseModel):
    id: str = Field(..., description="ID dùng ở trường `style` khi tạo audio.", examples=["tin_tuc"])
    label: str = Field(..., description="Tên hiển thị.", examples=["Tin tức (news)"])
    is_default: bool = Field(default=False)


class StylesResponse(BaseModel):
    count: int
    default_style: str
    styles: List[StyleInfo]


class HealthResponse(BaseModel):
    """Trạng thái tổng thể — mọi thứ đã sẵn sàng chưa."""

    status: str = Field(..., description="ok = sẵn sàng phục vụ; loading = đang tải model; error.", examples=["ok"])
    model_loaded: bool = Field(..., description="Model đã nạp vào RAM/VRAM chưa.")
    backend: Optional[str] = Field(default=None, description="pytorch (GPU) hoặc onnx (CPU).")
    device: Optional[str] = Field(default=None, description="cuda / cpu.", examples=["cuda"])
    backbone_repo: Optional[str] = Field(default=None, examples=["pnnbao-ump/VieNeu-TTS-v3-Turbo"])
    sample_rate: Optional[int] = Field(default=None, description="Tần số lấy mẫu audio xuất ra (Hz).", examples=[48000])
    num_voices: int = Field(default=0, description="Số giọng preset đã nạp.")
    ui_mounted: bool = Field(default=False, description="Có gắn kèm giao diện Gradio ở `/` không.")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Mô tả lỗi.")
