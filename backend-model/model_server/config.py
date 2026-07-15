"""Load backend-model/.env into the environment (once, at import).

Import this module before any ``os.getenv`` reads so a bare ``uv run`` picks up
R2 creds / MODEL_API_KEY / storage config without exporting anything. Values
already in the real environment win (``setdefault``), so Docker ``environment:``
overrides the file.
"""
from __future__ import annotations

import os
from pathlib import Path

# backend-model/  (two levels up from model_server/config.py)
_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv(_ROOT / ".env")
