"""jobs_repo — CRUD bảng ``jobs`` (persist job TTS). NƠI DUY NHẤT đọc/ghi jobs.

Job sống trong RAM khi đang chạy (JobManager, có cancel event), nhưng được SYNC
xuống DB ở mỗi cập nhật → sống qua restart + tra được lịch sử theo ``user_ref``.
Các field không-serialize (``voice_record`` numpy, ``cancel`` Event) KHÔNG lưu DB.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from ..db import connect

logger = logging.getLogger("Vieneu.JobsRepo")

# Các cột persist (khớp thứ tự trong save()).
_COLS = ("id", "user_ref", "text", "voice_id", "style", "temperature", "max_chars",
         "mode", "status", "progress", "done_chunks", "total_chunks", "audio_key",
         "audio_url", "duration_sec", "elapsed_sec", "sample_rate", "instance_id",
         "error")


def save(job) -> None:
    """UPSERT 1 job từ object RAM xuống DB (gọi ở create + mỗi lần touch)."""
    vals = (
        job.id, getattr(job, "user_ref", "default"), job.text, job.voice, job.style,
        job.temperature, job.max_chars, job.mode, job.status, job.progress,
        job.done_chunks, job.total_chunks, job.audio_key, job.audio_url,
        job.duration_sec, job.elapsed_sec, job.sample_rate,
        getattr(job, "instance_id", None), job.error,
    )
    with connect() as conn:
        conn.execute(
            """INSERT INTO jobs
               (id,user_ref,text,voice_id,style,temperature,max_chars,mode,status,
                progress,done_chunks,total_chunks,audio_key,audio_url,duration_sec,
                elapsed_sec,sample_rate,instance_id,error,updated_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
               ON CONFLICT(id) DO UPDATE SET
                 status=excluded.status, progress=excluded.progress,
                 done_chunks=excluded.done_chunks, total_chunks=excluded.total_chunks,
                 audio_key=excluded.audio_key, audio_url=excluded.audio_url,
                 duration_sec=excluded.duration_sec, elapsed_sec=excluded.elapsed_sec,
                 sample_rate=excluded.sample_rate, instance_id=excluded.instance_id,
                 error=excluded.error, updated_at=now()""",
            vals,
        )


def get(job_id: str) -> Optional[dict]:
    """1 job (dict thuần) từ DB — dùng khi job không còn trong RAM (sau restart)."""
    with connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = %s", (job_id,)).fetchone()
    return dict(row) if row else None


def list_for_user(user_ref: str, limit: int = 100) -> List[dict]:
    """Lịch sử job của 1 user, mới nhất trước."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE user_ref = %s ORDER BY created_at DESC LIMIT %s",
            (user_ref, limit)).fetchall()
    return [dict(r) for r in rows]


def mark_interrupted() -> int:
    """Job còn queued/running lúc khởi động = mồ côi từ lần chạy trước (RAM đã mất)
    → đánh dấu error. Gọi 1 lần lúc startup. Trả số job bị đánh dấu."""
    with connect() as conn:
        cur = conn.execute(
            """UPDATE jobs SET status='error', error='server restarted', updated_at=now()
               WHERE status IN ('queued','running')""")
    return cur.rowcount


def active_count() -> int:
    with connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) c FROM jobs WHERE status IN ('queued','running')"
        ).fetchone()["c"]
