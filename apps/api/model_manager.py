"""Loads and holds ONE ``Vieneu`` v3 Turbo instance for the whole API process.

The device is auto-detected: a CUDA machine (like the dev laptop) runs the
PyTorch engine; a CPU-only host (like the Proxmox CT) runs the torch-free ONNX
engine. Loading is guarded by a lock so concurrent requests can't trigger two
loads. Synthesis itself is serialized (the engine is not re-entrant) — fine for
a single-GPU / CPU box.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Vieneu.API")

# Reading styles kept in sync with schemas.STYLE_CHOICES.
from .schemas import STYLE_CHOICES, DEFAULT_STYLE  # noqa: E402


class ModelManager:
    """Thread-safe holder for a single ``Vieneu`` instance."""

    def __init__(self) -> None:
        self._tts = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._backend: Optional[str] = None
        self._device: Optional[str] = None
        self._backbone_repo: str = os.getenv(
            "VIENEU_BACKBONE_REPO", "pnnbao-ump/VieNeu-TTS-v3-Turbo"
        )
        self._load_error: Optional[str] = None

    # ── State introspection (for /health) ───────────────────────────────────
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
    def backbone_repo(self) -> str:
        return self._backbone_repo

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def sample_rate(self) -> Optional[int]:
        return getattr(self._tts, "sample_rate", None) if self._tts else None

    # ── Loading ──────────────────────────────────────────────────────────────
    def load(self) -> None:
        """Load the model once (idempotent). Raises on failure."""
        if self._tts is not None:
            return
        with self._load_lock:
            if self._tts is not None:  # re-check inside lock
                return
            t0 = time.time()
            logger.info("⏳ API: loading VieNeu-TTS v3 Turbo (device=auto)...")
            try:
                from vieneu import Vieneu

                tts = Vieneu(mode="v3turbo", backbone_repo=self._backbone_repo, device="auto")
                self._backend = getattr(tts, "backend", None)
                # Resolve the concrete device string for the health report.
                self._device = self._detect_device(tts)
                self._tts = tts
                self._load_error = None
                logger.info(
                    "✅ API: model ready in %.1fs (backend=%s, device=%s, voices=%d)",
                    time.time() - t0, self._backend, self._device, len(self.list_voices()),
                )
            except Exception as e:  # surface a clean message to /health
                self._load_error = f"{type(e).__name__}: {e}"
                logger.exception("❌ API: model load failed")
                raise

    @staticmethod
    def _detect_device(tts) -> str:
        engine = getattr(tts, "engine", None)
        dev = getattr(engine, "device", None)
        if dev is not None:
            return str(getattr(dev, "type", dev))
        # ONNX engine has no torch device → it's CPU.
        return "cpu" if getattr(tts, "backend", None) == "onnx" else "unknown"

    # ── Voices / styles ──────────────────────────────────────────────────────
    def list_voices(self) -> List[Tuple[str, str]]:
        """Return ``[(label, id), ...]`` for preset voices (empty if not loaded)."""
        if self._tts is None:
            return []
        return self._tts.list_preset_voices()

    def default_voice(self) -> Optional[str]:
        return getattr(self._tts, "_default_voice", None) if self._tts else None

    def voice_meta(self, voice_id: str) -> dict:
        """Best-effort metadata (gender) for a voice id."""
        if self._tts is None:
            return {}
        return self._tts._preset_voices.get(voice_id, {}) or {}

    @staticmethod
    def styles() -> List[Tuple[str, str]]:
        return list(STYLE_CHOICES.items())

    # ── Synthesis ────────────────────────────────────────────────────────────
    def synthesize(
        self,
        text: str,
        voice: Optional[str],
        style: str,
        temperature: float,
        max_chars: int,
        apply_watermark: bool,
    ) -> Tuple[np.ndarray, int, float]:
        """Run inference. Returns ``(wav, sample_rate, elapsed_seconds)``.

        Serialized via ``_infer_lock`` — the engine is single-session.
        """
        if self._tts is None:
            raise RuntimeError("Model chưa được nạp.")
        style = style if style in STYLE_CHOICES else DEFAULT_STYLE
        with self._infer_lock:
            t0 = time.time()
            wav = self._tts.infer(
                text,
                voice=voice,
                style=style,
                temperature=temperature,
                max_chars=max_chars,
                apply_watermark=apply_watermark,
            )
            elapsed = time.time() - t0
        sr = int(self.sample_rate or 48000)
        return wav, sr, elapsed


# Module-level singleton shared by all routes.
manager = ModelManager()
