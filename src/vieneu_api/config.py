"""Runtime settings, all overridable via environment variables.

Paths default under a ``data/`` dir at the repo root so a bare checkout works
without extra setup; Docker overrides them to volume-mounted locations.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repo root = three levels up from this file (src/vieneu_api/config.py).
_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    p = Path(os.getenv(name, str(default)))
    p.mkdir(parents=True, exist_ok=True)
    return p


class Settings:
    # Postgres (docker container vieneu-pg on 5433 by default).
    DATABASE_URL: str = os.getenv(
        "VIENEU_DATABASE_URL",
        "postgresql+psycopg://vieneu:vieneu@127.0.0.1:5433/vieneu",
    )

    # Where generated audio + uploaded/enrolled voice clips are stored on disk.
    DATA_DIR: Path = _env_path("VIENEU_DATA_DIR", _ROOT / "data")
    AUDIO_DIR: Path = _env_path("VIENEU_AUDIO_DIR", _ROOT / "data" / "audio")
    VOICE_UPLOAD_DIR: Path = _env_path("VIENEU_VOICE_UPLOAD_DIR", _ROOT / "data" / "voices")

    # Persisted custom voices (speaker_emb + codes) so they survive a restart.
    CUSTOM_VOICES_PATH: Path = Path(
        os.getenv("VIENEU_CUSTOM_VOICES_PATH", str(_ROOT / "data" / "custom_voices.json"))
    )

    # Model.
    BACKBONE_REPO: str = os.getenv("VIENEU_BACKBONE_REPO", "pnnbao-ump/VieNeu-TTS-v3-Turbo")
    EAGER_LOAD: bool = os.getenv("VIENEU_API_EAGER_LOAD", "1") == "1"

    # Execution mode:
    #   "local"  → BE loads the model in-process (single machine, PA1).
    #   "remote" → BE has NO model; it calls the GPU model-server over HTTP (PA3).
    MODE: str = os.getenv("VIENEU_MODE", "local").lower()
    MODEL_SERVER_URL: str = os.getenv("MODEL_SERVER_URL", "http://127.0.0.1:9000")
    MODEL_API_KEY: str = os.getenv("MODEL_API_KEY", "dev-secret-key")

    # Chunking / synthesis defaults.
    MAX_CHARS_PER_CHUNK: int = int(os.getenv("VIENEU_MAX_CHARS", "256"))
    MAX_NEW_FRAMES: int = int(os.getenv("VIENEU_MAX_NEW_FRAMES", "300"))

    # Server.
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
    MOUNT_DEMO: bool = os.getenv("VIENEU_MOUNT_DEMO", "1") == "1"


settings = Settings()
