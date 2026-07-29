"""Hot model wrapper — nạp code gốc tác giả (``vieneu`` SDK) 1 lần, in-process.

Đây là 'đầu IO' của server: mọi thứ liên quan model đều đi qua module này. CPU mode
gọi thẳng vào đây (không HTTP). device=auto → CUDA/PyTorch nếu có, else ONNX/CPU.
Lazy-load: chỉ nạp khi gọi ``load()`` lần đầu (hoặc eager ở startup).
"""
from __future__ import annotations

import logging
import threading
from typing import List, Optional, Tuple

import numpy as np

from .config import settings

logger = logging.getLogger("Vieneu.Engine")

STYLE_CHOICES = {
    "tu_nhien": "Tự nhiên (conversational)",
    "tin_tuc": "Tin tức (news)",
    "doc_truyen": "Kể chuyện (storytelling)",
}
DEFAULT_STYLE = "tu_nhien"


class Engine:
    """Bọc ``vieneu.Vieneu(mode='v3turbo')`` với khóa nạp/suy luận an toàn luồng."""

    def __init__(self) -> None:
        self._tts = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._backend: Optional[str] = None
        self._device: Optional[str] = None
        self._backbone = settings.BACKBONE_REPO
        self._load_error: Optional[str] = None

    # ── Trạng thái ────────────────────────────────────────────────────────────
    @property
    def loaded(self) -> bool:
        return self._tts is not None

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    @property
    def device(self) -> Optional[str]:
        return self._device

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def sample_rate(self) -> int:
        return int(getattr(self._tts, "sample_rate", 48000)) if self._tts else 48000

    @property
    def tts(self):
        return self._tts

    # ── Nạp model (gọi vào gốc tác giả) ──────────────────────────────────────
    def load(self) -> None:
        if self._tts is not None:
            return
        with self._load_lock:
            if self._tts is not None:
                return
            logger.info("⏳ Nạp VieNeu-TTS v3 Turbo (device=%s)...", settings.MODEL_DEVICE)
            try:
                from vieneu import Vieneu
                tts = Vieneu(mode="v3turbo", backbone_repo=self._backbone,
                             device=settings.MODEL_DEVICE)
                self._backend = getattr(tts, "backend", None)
                eng = getattr(tts, "engine", None)
                dev = getattr(eng, "device", None)
                self._device = str(getattr(dev, "type", dev)) if dev is not None else (
                    "cpu" if self._backend == "onnx" else "unknown")
                self._tts = tts
                self._load_error = None
                n_preset = len(getattr(tts, "_preset_voices", {}) or {})
                logger.info("✅ Model hot: backend=%s device=%s preset(SDK)=%d",
                            self._backend, self._device, n_preset)
            except Exception as e:
                self._load_error = f"{type(e).__name__}: {e}"
                logger.exception("❌ Nạp model thất bại")
                raise

    # ── Preset ở RAM (giọng hệ thống, KHÔNG vào DB) ──────────────────────────
    # 10 giọng preset do SDK nạp sẵn (bundle của model). Dùng chung cho MỌI user,
    # bất biến, không lưu DB. DB chỉ chứa giọng custom user clone.
    def _presets(self) -> dict:
        """Dict preset của SDK (an toàn khi _tts chưa/không phải Vieneu thật)."""
        return getattr(self._tts, "_preset_voices", {}) or {}

    def default_preset_id(self) -> Optional[str]:
        return getattr(self._tts, "_default_voice", None) if self._tts else None

    def has_preset(self, vid: str) -> bool:
        return vid in self._presets()

    def _preset_to_record(self, vid: str, v: dict) -> dict:
        return {
            "id": vid, "user_ref": None,
            "description": v.get("description", ""), "gender": v.get("gender", ""),
            "region": v.get("region", ""), "style": v.get("style", DEFAULT_STYLE),
            "source": "preset", "is_default": (vid == self.default_preset_id()),
            "speaker_emb": v.get("speaker_emb"), "codes": v.get("codes"),
        }

    def get_preset_record(self, vid: str) -> Optional[dict]:
        """1 giọng preset → record runtime (numpy) như voices_repo.get trả ra.
        Nhờ đó preset và custom đi CÙNG 1 dạng record xuống synth (CPU/GPU)."""
        v = self._presets().get(vid)
        return self._preset_to_record(vid, v) if isinstance(v, dict) else None

    def list_preset_records(self) -> List[dict]:
        """Mọi giọng preset (record runtime) — để GET /voices ghép với custom."""
        return [self._preset_to_record(n, v)
                for n, v in self._presets().items() if isinstance(v, dict)]

    # ── Trích đặc trưng giọng từ audio (cho enroll) ──────────────────────────
    def extract_reference(self, ref_audio_path: str, *, denoise: bool = True):
        """Audio mẫu → ``(speaker_emb, codes)`` (numpy). KHÔNG lưu RAM — caller
        (voices service) tự ghi vào DB. Đây là phần 'cần model' của enroll."""
        with self._infer_lock:
            return self._tts.encode_reference(ref_audio_path, denoise=denoise)

    # ── Chunk + synth từng chunk (để báo % tiến độ) ──────────────────────────
    def split_chunks(self, text: str, max_chars: int):
        from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps
        return normalize_to_chunks_v3_with_gaps(text, max_chars=max_chars)

    def synth_chunk(self, chunk_text: str, voice_record: dict, style: str,
                    temperature: float) -> np.ndarray:
        """Synth THUẦN: nhận record giọng (dict có speaker_emb + codes) đã lấy từ
        DB. Engine KHÔNG còn tra catalog theo tên — mọi giọng (preset/custom) đều
        vào đây dưới cùng 1 dạng record, giống hệt đường GPU. CPU==GPU về input."""
        from vieneu_utils.phonemize_text import phonemize_text_with_emotions
        style = style if style in STYLE_CHOICES else DEFAULT_STYLE
        spk = voice_record.get("speaker_emb")
        ref = voice_record.get("codes")
        with self._infer_lock:
            ph = phonemize_text_with_emotions(chunk_text)
            return self._tts.engine.infer(
                phonemes=ph, speaker_emb=spk, ref_codes=ref, style=style,
                use_ref_codes=ref is not None, temperature=temperature,
                max_new_frames=settings.MAX_NEW_FRAMES)


engine = Engine()
