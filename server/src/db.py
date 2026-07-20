"""DB — kết nối PostgreSQL (psycopg v3 + connection pool).

Cấu hình qua ``DATABASE_URL`` trong ``.env`` (xem ``docker-compose.yaml``). Trong
Docker host là ``db`` (@db:5432); chạy tay ngoài Docker dùng ``localhost:7432``.
Dùng 1 pool dùng chung an toàn đa luồng (job chạy ở nhiều thread) — mượn/trả
connection qua ``with connect()``.

2 bảng, CẢ HAI đều có ``user_ref`` để mỗi người dùng có kho riêng:
* ``voices`` — giọng CUSTOM do user clone (preset KHÔNG vào đây, ở RAM engine).
              Khóa chính (user_ref, id) → 2 user được trùng tên giọng.
* ``jobs``   — job TTS (persist, sống qua restart, tra theo user_ref).

Xem ``repositories/voices_repo.py`` + ``repositories/jobs_repo.py`` cho CRUD.
Placeholder tham số dùng ``%s`` (psycopg), KHÔNG phải ``?``.
"""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager

from .config import settings

logger = logging.getLogger("Vieneu.DB")

_pool = None
_pool_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS voices (
    user_ref      TEXT NOT NULL DEFAULT 'default',
    id            TEXT NOT NULL,
    name          TEXT NOT NULL DEFAULT '',
    description   TEXT NOT NULL DEFAULT '',
    gender        TEXT NOT NULL DEFAULT '',
    region        TEXT NOT NULL DEFAULT '',
    default_style TEXT NOT NULL DEFAULT 'tu_nhien',
    speaker_emb   JSONB,              -- list[float] (192)
    codes         JSONB,              -- list[list[int]] hoặc null
    ref_audio_key TEXT,               -- (tùy chọn) key R2 audio gốc để re-enroll sau
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_ref, id)        -- mỗi user 1 kho; cho trùng tên giữa các user
);

CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,   -- uuid
    user_ref      TEXT NOT NULL DEFAULT 'default',
    text          TEXT NOT NULL DEFAULT '',
    voice_id      TEXT,               -- id giọng (preset hoặc custom của user)
    style         TEXT NOT NULL DEFAULT 'tu_nhien',
    temperature   REAL NOT NULL DEFAULT 0.8,
    max_chars     INTEGER NOT NULL DEFAULT 256,
    mode          TEXT NOT NULL DEFAULT 'cpu',     -- cpu | gpu
    status        TEXT NOT NULL DEFAULT 'queued',  -- queued|running|done|cancelled|error
    progress      REAL NOT NULL DEFAULT 0,
    done_chunks   INTEGER NOT NULL DEFAULT 0,
    total_chunks  INTEGER NOT NULL DEFAULT 0,
    audio_key     TEXT,
    audio_url     TEXT,
    duration_sec  REAL,
    elapsed_sec   REAL,
    sample_rate   INTEGER,
    instance_id   BIGINT,             -- (GPU) truy vết tiền
    error         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_ref, created_at);
CREATE INDEX IF NOT EXISTS idx_voices_user ON voices(user_ref, created_at);
"""


def _get_pool():
    """Pool dùng chung (lazy). psycopg_pool mở/giữ vài connection tái dùng."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                from psycopg_pool import ConnectionPool
                _pool = ConnectionPool(settings.DATABASE_URL, min_size=1, max_size=10,
                                       kwargs={"autocommit": False}, open=True)
    return _pool


@contextmanager
def connect():
    """Mượn 1 connection từ pool (row=dict), tự commit/rollback + trả về pool."""
    from psycopg.rows import dict_row
    pool = _get_pool()
    with pool.connection() as conn:      # tự trả về pool khi thoát with
        conn.row_factory = dict_row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def init_db() -> None:
    """Tạo bảng nếu chưa có (idempotent). Gọi 1 lần lúc khởi động."""
    with connect() as conn:
        conn.execute(_SCHEMA)
    # Che mật khẩu khi log.
    safe = settings.DATABASE_URL
    if "@" in safe:
        safe = safe.split("://", 1)[0] + "://***@" + safe.split("@", 1)[1]
    logger.info("🗃️  PostgreSQL sẵn sàng: %s", safe)
