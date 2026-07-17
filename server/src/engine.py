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
                logger.info("✅ Model hot: backend=%s device=%s voices=%d",
                            self._backend, self._device, len(self.list_voices()))
            except Exception as e:
                self._load_error = f"{type(e).__name__}: {e}"
                logger.exception("❌ Nạp model thất bại")
                raise

    # ── Catalog giọng ────────────────────────────────────────────────────────
    def list_voices(self) -> List[Tuple[str, str]]:
        return self._tts.list_preset_voices() if self._tts else []

    def default_voice(self) -> Optional[str]:
        return getattr(self._tts, "_default_voice", None) if self._tts else None

    def has_voice(self, vid: str) -> bool:
        return bool(self._tts) and vid in self._tts._preset_voices

    def voice_meta(self, vid: str) -> dict:
        return (self._tts._preset_voices.get(vid, {}) or {}) if self._tts else {}

    def export_voice(self, vid: str) -> Optional[dict]:
        """Serialize 1 giọng → dict JSON-safe (speaker_emb + codes dạng list).

        Dùng để 'lôi giọng ra ngoài': gửi sang máy GPU ephemeral (SCP JSON vài KB)
        hoặc lưu DB. Format y hệt ``save_voices`` cho 1 giọng nên nạp lại bằng
        ``infer(voice=<dict>)`` chạy ngay, KHÔNG cần audio gốc / enroll lại.
        """
        v = self._tts._preset_voices.get(vid) if self._tts else None
        if not isinstance(v, dict):
            return None
        emb, codes = v.get("speaker_emb"), v.get("codes")
        return {
            "description": v.get("description", ""),
            "gender": v.get("gender", ""),
            "style": v.get("style", DEFAULT_STYLE),
            "speaker_emb": None if emb is None
            else [round(float(x), 6) for x in np.asarray(emb).reshape(-1)],
            "codes": None if codes is None else np.asarray(codes, dtype=int).tolist(),
        }

    # ── Chunk + synth từng chunk (để báo % tiến độ) ──────────────────────────
    def split_chunks(self, text: str, max_chars: int):
        from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps
        return normalize_to_chunks_v3_with_gaps(text, max_chars=max_chars)

    def synth_chunk(self, chunk_text: str, voice, style: str, temperature: float) -> np.ndarray:
        from vieneu_utils.phonemize_text import phonemize_text_with_emotions
        style = style if style in STYLE_CHOICES else DEFAULT_STYLE
        with self._infer_lock:
            spk, ref = self._tts._resolve_ref(voice=voice, ref_audio=None,
                                              denoise=True, use_ref_codes=True)
            ph = phonemize_text_with_emotions(chunk_text)
            return self._tts.engine.infer(
                phonemes=ph, speaker_emb=spk, ref_codes=ref, style=style,
                use_ref_codes=True, temperature=temperature,
                max_new_frames=settings.MAX_NEW_FRAMES)

    # ── Voice enrollment (clone) ─────────────────────────────────────────────
    def enroll(self, name: str, ref_audio_path: str, *, denoise: bool = True,
               description: str = "", gender: str = "", style: str = DEFAULT_STYLE) -> None:
        with self._infer_lock:
            self._tts.add_voice(name, ref_audio_path, denoise=denoise,
                                description=description, gender=gender, style=style, save=False)
            # Đánh dấu giọng clone (chỉ có trong RAM VPS) để phân biệt với preset
            # bundle. Dùng ở: voices.py (source=custom) + gpu_vastai (serialize gửi
            # GPU). Chỉ là cờ RAM — save_voices/export_voice không serialize key này.
            v = self._tts._preset_voices.get(name)
            if isinstance(v, dict):
                v["_custom"] = True

    def remove(self, name: str) -> None:
        if self._tts:
            self._tts.remove_voice(name, save=False)


engine = Engine()
