"""createVoice — orchestrator cho API tạo TTS (điểm rẽ nhánh CPU/GPU).

Luồng: controller nhận request → gọi ``create()`` ở đây → check đủ điều kiện
(model sẵn sàng, voice tồn tại, ngưỡng ký tự cho GPU) → CHỐT mode → tạo job
(``jobs.manager``) → job worker gọi xuống ``cpu_onnx`` hoặc ``gpu_vastai``.

2 lớp chặn mode GPU: FE đã chặn 1 lớp; đây là lớp BE chặn thêm — GPU chỉ hợp lệ
khi context ≥ ``settings.GPU_MIN_WORDS`` TỪ.
"""
from __future__ import annotations

import logging
from typing import Optional

from ..config import settings
from ..engine import engine
from ..repositories import voices_repo as repo
from ..schemas import (
    DEFAULT_STYLE, MODE_AUTO, MODE_CPU, MODE_GPU, MODE_CHOICES, STYLE_CHOICES,
)
from .jobs import Job, manager

logger = logging.getLogger("Vieneu.CreateVoice")


class CreateError(Exception):
    """Lỗi nghiệp vụ khi tạo job (controller map sang HTTP status)."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def count_words(text: str) -> int:
    """Đếm SỐ TỪ (tách theo khoảng trắng) — dùng cho ngưỡng chọn GPU."""
    return len(text.split())


def resolve_mode(text: str, requested: str) -> str:
    """Chốt mode thực tế từ mode client yêu cầu + độ dài context.

    * requested='gpu' nhưng context < ngưỡng  → CHẶN (lỗi 422).
    * requested='auto'                         → gpu nếu đủ dài, else cpu.
    * requested='cpu'                          → luôn cpu.
    """
    requested = (requested or MODE_AUTO).lower()
    if requested not in MODE_CHOICES:
        raise CreateError(422, f"mode '{requested}' không hợp lệ. Chọn: {', '.join(MODE_CHOICES)}.")

    n = count_words(text)
    long_enough = n >= settings.GPU_MIN_WORDS

    if requested == MODE_GPU:
        if not long_enough:
            raise CreateError(
                422,
                f"GPU mode cần context ≥ {settings.GPU_MIN_WORDS} từ; hiện có {n}. "
                f"Dùng CPU mode cho văn bản ngắn.",
            )
        return MODE_GPU

    if requested == MODE_AUTO:
        return MODE_GPU if long_enough else MODE_CPU

    return MODE_CPU


def resolve_voice(user_ref: str, voice: Optional[str]) -> dict:
    """Chốt record giọng: PRESET (RAM, dùng chung) TRƯỚC, else CUSTOM của user (DB).

    Bỏ trống voice → giọng preset mặc định. Ném CreateError(422) nếu không thấy.
    """
    voice_id = voice or engine.default_preset_id()
    if not voice_id:
        raise CreateError(422, "Chưa có giọng nào khả dụng.")
    # 1) preset (RAM) — dùng chung mọi user
    record = engine.get_preset_record(voice_id)
    # 2) custom của chính user (DB)
    if record is None:
        record = repo.get(user_ref, voice_id)
    if record is None:
        raise CreateError(422, f"Voice '{voice_id}' không tồn tại. Xem GET /api/v1/voices.")
    return record


def create(text: str, voice: Optional[str], style: str, temperature: float,
           max_chars: int, mode: str = MODE_AUTO, user_ref: str = "default") -> Job:
    """Kiểm tra điều kiện, chốt mode, tạo job. Trả về Job (đã có id + mode)."""
    # 1. Model phải sẵn sàng (CPU in-process synth cần model nạp sẵn).
    if not engine.loaded:
        raise CreateError(503, "Model chưa sẵn sàng. Xem GET /api/v1/health.")

    # 2. Chốt giọng: preset (RAM) hoặc custom của user (DB).
    record = resolve_voice(user_ref, voice)
    voice_id = record["id"]

    # 3. Style hợp lệ (fallback mặc định nếu sai).
    style = style if style in STYLE_CHOICES else DEFAULT_STYLE

    # 4. Chốt mode (lớp chặn GPU thứ 2 ở BE).
    resolved = resolve_mode(text, mode)

    # 5. Tạo job async, đính kèm RECORD giọng (dict emb+codes) để worker (cpu/gpu)
    #    dùng thẳng — không phải tra lại catalog. CPU==GPU cùng 1 record.
    job = manager.create(text, voice_id, style, temperature, max_chars,
                         mode=resolved, voice_record=record, user_ref=user_ref)
    logger.info("🎬 tạo job %s user=%s mode=%s (yêu cầu=%s, %d từ) voice=%s src=%s",
                job.id[:8], user_ref, resolved, mode, count_words(text), voice_id,
                record.get("source"))
    return job
