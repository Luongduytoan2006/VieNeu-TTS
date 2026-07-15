"""Holds ONE hot ``Vieneu`` v3 Turbo instance for the whole process.

Loaded once at startup (see app lifespan) so requests never pay a cold load.
Exposes the low-level pieces the job worker needs: per-chunk synthesis (so it can
report progress and honor cancellation between chunks), voice enrollment, and the
voice/style catalog. Reuses the existing SDK utilities rather than reimplementing.
"""
from __future__ import annotations

import logging
import threading
from typing import List, Optional, Tuple

import numpy as np

from ..config import settings
from ..schemas import DEFAULT_STYLE, STYLE_CHOICES

logger = logging.getLogger("Vieneu.API.tts")


class TTSService:
    def __init__(self) -> None:
        self._tts = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()   # engine is single-session
        self._backend: Optional[str] = None
        self._device: Optional[str] = None
        self._load_error: Optional[str] = None

    # ── State ────────────────────────────────────────────────────────────────
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
    def sample_rate(self) -> Optional[int]:
        return getattr(self._tts, "sample_rate", None) if self._tts else None

    @property
    def tts(self):
        return self._tts

    # ── Loading (hot) ────────────────────────────────────────────────────────
    def load(self) -> None:
        if self._tts is not None:
            return
        with self._load_lock:
            if self._tts is not None:
                return
            logger.info("⏳ Loading VieNeu-TTS v3 Turbo (device=auto)...")
            try:
                from vieneu import Vieneu

                tts = Vieneu(mode="v3turbo", backbone_repo=settings.BACKBONE_REPO, device="auto")
                self._backend = getattr(tts, "backend", None)
                self._device = self._detect_device(tts)
                self._tts = tts
                self._load_error = None
                logger.info("✅ Model hot: backend=%s device=%s voices=%d",
                            self._backend, self._device, len(self.list_voices()))
            except Exception as e:
                self._load_error = f"{type(e).__name__}: {e}"
                logger.exception("❌ Model load failed")
                raise

    @staticmethod
    def _detect_device(tts) -> str:
        engine = getattr(tts, "engine", None)
        dev = getattr(engine, "device", None)
        if dev is not None:
            return str(getattr(dev, "type", dev))
        return "cpu" if getattr(tts, "backend", None) == "onnx" else "unknown"

    # ── Catalog ──────────────────────────────────────────────────────────────
    def list_voices(self) -> List[Tuple[str, str]]:
        if self._tts is None:
            return []
        return self._tts.list_preset_voices()

    def default_voice(self) -> Optional[str]:
        return getattr(self._tts, "_default_voice", None) if self._tts else None

    def voice_meta(self, voice_id: str) -> dict:
        if self._tts is None:
            return {}
        return self._tts._preset_voices.get(voice_id, {}) or {}

    def has_voice(self, voice_id: str) -> bool:
        return bool(self._tts) and voice_id in self._tts._preset_voices

    @staticmethod
    def styles() -> List[Tuple[str, str]]:
        return list(STYLE_CHOICES.items())

    # ── Chunking (reuse SDK utils) ───────────────────────────────────────────
    def split_chunks(self, text: str, max_chars: int) -> Tuple[List[str], list]:
        from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps
        return normalize_to_chunks_v3_with_gaps(text, max_chars=max_chars)

    # ── Per-chunk synthesis (for progress + cancellation) ────────────────────
    def synth_chunk(self, chunk_text: str, voice: Optional[str], style: str,
                    temperature: float) -> np.ndarray:
        """Synthesize ONE chunk → float32 waveform. Serialized (engine is 1-session).

        Resolves the voice via the SDK (preset or enrolled custom), phonemizes the
        chunk, then runs the engine directly so we control it per-chunk.
        """
        if self._tts is None:
            raise RuntimeError("Model chưa nạp.")
        from vieneu_utils.phonemize_text import phonemize_text_with_emotions

        style = style if style in STYLE_CHOICES else DEFAULT_STYLE
        with self._infer_lock:
            speaker_emb, ref_codes = self._tts._resolve_ref(
                voice=voice, ref_audio=None, denoise=True, use_ref_codes=True
            )
            ph = phonemize_text_with_emotions(chunk_text)
            wav = self._tts.engine.infer(
                phonemes=ph, speaker_emb=speaker_emb, ref_codes=ref_codes,
                style=style, use_ref_codes=True,
                temperature=temperature, max_new_frames=settings.MAX_NEW_FRAMES,
            )
        return wav

    # ── Voice enrollment (clone from an uploaded clip) ───────────────────────
    def enroll_voice(self, name: str, ref_audio_path: str, *, denoise: bool = True,
                     description: str = "", gender: str = "", style: str = DEFAULT_STYLE) -> str:
        """Analyze an audio clip → add a reusable voice (speaker_emb + codes)."""
        if self._tts is None:
            raise RuntimeError("Model chưa nạp.")
        with self._infer_lock:
            return self._tts.add_voice(
                name, ref_audio_path, denoise=denoise, description=description,
                gender=gender, style=style, save=False,
            )

    def remove_voice(self, name: str) -> None:
        if self._tts is not None:
            self._tts.remove_voice(name, save=False)

    def save_custom_voices(self, path) -> None:
        if self._tts is not None:
            self._tts.save_voices(path)

    def load_custom_voices(self, path) -> int:
        """Merge persisted custom voices into the hot model. Returns count added."""
        import json
        from pathlib import Path
        p = Path(path)
        if self._tts is None or not p.exists():
            return 0
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return 0
        added = 0
        for name, v in (data.get("presets") or {}).items():
            emb = v.get("speaker_emb")
            codes = v.get("codes")
            self._tts._preset_voices[name] = {
                "description": v.get("description", ""),
                "gender": v.get("gender", ""),
                "style": v.get("style", DEFAULT_STYLE),
                "speaker_emb": np.asarray(emb, dtype=np.float32) if emb is not None else None,
                "codes": np.asarray(codes, dtype=np.int64) if codes is not None else None,
            }
            added += 1
        return added


service = TTSService()
