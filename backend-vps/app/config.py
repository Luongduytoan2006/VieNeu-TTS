"""Runtime settings for backend-vps, all overridable via environment variables.

Loaded once at import. A local ``.env`` (repo backend-vps/.env) is read first so a
bare ``uv run`` works without exporting anything.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Load backend-vps/.env (simple, dependency-free) ──────────────────────────
_ROOT = Path(__file__).resolve().parents[1]   # backend-vps/


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_dotenv(_ROOT / ".env")


class Settings:
    # ── Public API server ────────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "7862"))
    API_PREFIX: str = "/api/v1"
    FORCE_HTTPS: bool = os.getenv("VIENEU_FORCE_HTTPS", "0") == "1"
    MOUNT_DEMO: bool = os.getenv("VIENEU_MOUNT_DEMO", "1") == "1"

    # ── GPU model-server (backend-model) it forwards to ──────────────────────
    MODEL_SERVER_URL: str = os.getenv("MODEL_SERVER_URL", "http://127.0.0.1:9000")
    MODEL_API_KEY: str = os.getenv("MODEL_API_KEY", "dev-secret-key")

    # ── Database (durable jobs) ──────────────────────────────────────────────
    DATABASE_URL: str = os.getenv(
        "VIENEU_DATABASE_URL",
        "postgresql+psycopg://vieneu:vieneu@127.0.0.1:5433/vieneu",
    )

    # ── Synthesis defaults (forwarded to the model-server) ───────────────────
    BACKBONE_REPO: str = os.getenv("VIENEU_BACKBONE_REPO", "pnnbao-ump/VieNeu-TTS-v3-Turbo")
    MAX_CHARS_PER_CHUNK: int = int(os.getenv("VIENEU_MAX_CHARS", "256"))

    # ── Job polling cadence (BE mirrors model-server progress) ───────────────
    POLL_INTERVAL_SEC: float = float(os.getenv("VIENEU_POLL_INTERVAL", "1.2"))


settings = Settings()
