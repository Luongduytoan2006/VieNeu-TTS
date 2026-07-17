"""DB — kết nối SQLite (stdlib ``sqlite3``, chạy với chỉ mỗi ``uv``, hợp VPS 1 máy).

Đây là nơi mở/khởi tạo cơ sở dữ liệu. Mặc định file ``server/data/vieneu.db``
(đổi qua ``VIENEU_DB_PATH``). Mỗi thao tác mở 1 connection riêng (sqlite mở rất
nhẹ) để an toàn khi job chạy ở nhiều thread — không giữ connection dùng chung.

Bảng: ``voices`` (nguồn sự thật cho giọng, cả preset lẫn custom). Xem
``repositories/voices_repo.py`` cho CRUD.
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .config import settings

logger = logging.getLogger("Vieneu.DB")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS voices (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL DEFAULT '',
    description   TEXT NOT NULL DEFAULT '',
    gender        TEXT NOT NULL DEFAULT '',
    region        TEXT NOT NULL DEFAULT '',
    default_style TEXT NOT NULL DEFAULT 'tu_nhien',
    speaker_emb   TEXT,               -- JSON: list[float] (192)
    codes         TEXT,               -- JSON: list[list[int]] hoặc null
    source        TEXT NOT NULL DEFAULT 'custom',   -- preset | custom
    owner_id      TEXT,               -- để trống giờ; gắn user sau
    ref_audio_key TEXT,               -- (tùy chọn) key R2 audio gốc để re-enroll sau
    is_default    INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
"""


def db_path() -> Path:
    return Path(settings.DB_PATH)


@contextmanager
def connect():
    """Mở 1 connection (row=dict-like) + tự commit/close. Dùng theo ``with``."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Tạo bảng nếu chưa có (idempotent). Gọi 1 lần lúc khởi động."""
    with connect() as conn:
        conn.executescript(_SCHEMA)
    logger.info("🗃️  SQLite sẵn sàng: %s", db_path())
