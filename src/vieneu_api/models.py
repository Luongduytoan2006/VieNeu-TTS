"""ORM models: Job (a TTS request) and Voice (a cloned/custom voice)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


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

    # Result.
    audio_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # In PA3 (split BE/GPU) audio lives on the model-server's storage; the BE keeps
    # a URL/key to it instead of a local path. Either audio_path (local) or
    # audio_url (remote storage, e.g. MinIO/R2/local file-server) is set.
    audio_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    remote_job_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    elapsed_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    sample_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class Voice(Base):
    __tablename__ = "voices"

    # id = the voice name used at /tts (unique key the SDK resolves by).
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    description: Mapped[str] = mapped_column(Text, default="")
    gender: Mapped[str | None] = mapped_column(String(16), nullable=True)
    style: Mapped[str] = mapped_column(String(32), default="tu_nhien")
    source: Mapped[str] = mapped_column(String(16), default="custom")  # custom | preset
    ref_audio_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
