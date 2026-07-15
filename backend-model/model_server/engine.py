"""Hot model wrapper for the model-server (reuses the v3 Turbo SDK).

Same idea as the BE's tts_service, but this is the process that actually owns the
GPU/CPU model. device=auto → GPU (PyTorch) when available, else ONNX/CPU.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Vieneu.ModelServer.engine")

STYLE_CHOICES = {
    "tu_nhien": "Tự nhiên (conversational)",
    "tin_tuc": "Tin tức (news)",
    "doc_truyen": "Kể chuyện (storytelling)",
}
DEFAULT_STYLE = "tu_nhien"


class Engine:
    def __init__(self) -> None:
        self._tts = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._backend: Optional[str] = None
        self._device: Optional[str] = None
        self._backbone = os.getenv("VIENEU_BACKBONE_REPO", "pnnbao-ump/VieNeu-TTS-v3-Turbo")
        self._load_error: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self._tts is not None

    @property
    def backend(self):
        return self._backend

    @property
    def device(self):
        return self._device

    @property
    def load_error(self):
        return self._load_error

    @property
    def sample_rate(self) -> int:
        return int(getattr(self._tts, "sample_rate", 48000)) if self._tts else 48000

    @property
    def tts(self):
        return self._tts

    def load(self) -> None:
        if self._tts is not None:
            return
        with self._load_lock:
            if self._tts is not None:
                return
            logger.info("⏳ model-server loading VieNeu-TTS v3 Turbo (device=auto)...")
            try:
                from vieneu import Vieneu
                tts = Vieneu(mode="v3turbo", backbone_repo=self._backbone, device="auto")
                self._backend = getattr(tts, "backend", None)
                eng = getattr(tts, "engine", None)
                dev = getattr(eng, "device", None)
                self._device = str(getattr(dev, "type", dev)) if dev is not None else (
                    "cpu" if self._backend == "onnx" else "unknown")
                self._tts = tts
                self._load_error = None
                logger.info("✅ model-server hot: backend=%s device=%s voices=%d",
                            self._backend, self._device, len(self.list_voices()))
            except Exception as e:
                self._load_error = f"{type(e).__name__}: {e}"
                logger.exception("❌ model-server load failed")
                raise

    # catalog
    def list_voices(self) -> List[Tuple[str, str]]:
        return self._tts.list_preset_voices() if self._tts else []

    def default_voice(self):
        return getattr(self._tts, "_default_voice", None) if self._tts else None

    def has_voice(self, vid: str) -> bool:
        return bool(self._tts) and vid in self._tts._preset_voices

    def voice_meta(self, vid: str) -> dict:
        return (self._tts._preset_voices.get(vid, {}) or {}) if self._tts else {}

    # chunking + per-chunk synth (for progress)
    def split_chunks(self, text: str, max_chars: int):
        from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps
        return normalize_to_chunks_v3_with_gaps(text, max_chars=max_chars)

    def synth_chunk(self, chunk_text: str, voice, style: str, temperature: float) -> np.ndarray:
        from vieneu_utils.phonemize_text import phonemize_text_with_emotions
        style = style if style in STYLE_CHOICES else DEFAULT_STYLE
        with self._infer_lock:
            spk, ref = self._tts._resolve_ref(voice=voice, ref_audio=None, denoise=True, use_ref_codes=True)
            ph = phonemize_text_with_emotions(chunk_text)
            return self._tts.engine.infer(
                phonemes=ph, speaker_emb=spk, ref_codes=ref, style=style,
                use_ref_codes=True, temperature=temperature,
                max_new_frames=int(os.getenv("VIENEU_MAX_NEW_FRAMES", "300")))

    # voice enrollment
    def enroll(self, name: str, ref_audio_path: str, *, denoise=True, description="",
               gender="", style=DEFAULT_STYLE) -> None:
        with self._infer_lock:
            self._tts.add_voice(name, ref_audio_path, denoise=denoise,
                                description=description, gender=gender, style=style, save=False)

    def remove(self, name: str) -> None:
        if self._tts:
            self._tts.remove_voice(name, save=False)


engine = Engine()
