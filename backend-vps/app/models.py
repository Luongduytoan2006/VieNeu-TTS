"""ORM models. backend-vps is remote-only, so it persists just the durable Job.

The audio itself lives on the model-server's storage (R2); the Job keeps a URL to
it (``audio_url``) plus the model-server's own job id (``remote_job_id``).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# Job lifecycle.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_CANCELLED = "cancelled"
STATUS_ERROR = "error"
ACTIVE_STATUSES = {STATUS_QUEUED, STATUS_RUNNING}


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    status: Mapped[str] = mapped_column(String(16), default=STATUS_QUEUED, index=True)

    text: Mapped[str] = mapped_column(Text)
    voice: Mapped[str | None] = mapped_column(String(128), nullable=True)
    style: Mapped[str] = mapped_column(String(32), default="tu_nhien")
    temperature: Mapped[float] = mapped_column(Float, default=0.8)

    # Progress: chunks are the unit of work; percent = done/total * 100.
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    done_chunks: Mapped[int] = mapped_column(Integer, default=0)
    progress: Mapped[float] = mapped_column(Float, default=0.0)

    # Result: audio lives on the model-server storage (R2); we keep its URL + id.
    audio_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    remote_job_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    elapsed_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    sample_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
