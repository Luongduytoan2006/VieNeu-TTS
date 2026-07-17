"""Quản lý job TTS async — RAM (đang chạy) + persist DB (sống qua restart).

Job có uuid, chạy trong 1 thread riêng, stream % tiến độ. Việc synth thật đẩy
xuống backend theo mode: ``cpu_onnx`` (in-process) hoặc ``gpu_vastai`` (Vast.ai).
Có cancel Event để hủy giữa chừng.

Persist: mỗi ``touch()`` sync job xuống ``jobs_repo`` (DB). ``get`` đọc RAM trước
(job đang sống, có tiến độ realtime), không có thì đọc DB (job cũ sau restart).
Mọi job đều gắn ``user_ref`` để tra theo người dùng.
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

from ..repositories import jobs_repo
from ..schemas import DEFAULT_STYLE, MODE_CPU

logger = logging.getLogger("Vieneu.Jobs")

QUEUED, RUNNING, DONE, CANCELLED, ERROR = "queued", "running", "done", "cancelled", "error"
ACTIVE_STATUSES = {QUEUED, RUNNING}


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Job:
    id: str
    text: str
    voice: Optional[str]
    style: str
    temperature: float
    max_chars: int
    user_ref: str = "default"
    mode: str = MODE_CPU
    # Record giọng (dict: speaker_emb+codes numpy) đã lấy từ DB/preset lúc tạo job.
    # Worker cpu/gpu dùng thẳng — không tra lại catalog. KHÔNG persist (numpy).
    voice_record: Optional[dict] = None
    status: str = QUEUED
    total_chunks: int = 0
    done_chunks: int = 0
    progress: float = 0.0
    audio_url: Optional[str] = None
    audio_key: Optional[str] = None
    duration_sec: Optional[float] = None
    elapsed_sec: Optional[float] = None
    sample_rate: Optional[int] = None
    instance_id: Optional[int] = None       # (GPU) truy vết tiền
    error: Optional[str] = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    cancel: threading.Event = field(default_factory=threading.Event)

    def touch(self) -> None:
        """Cập nhật mốc thời gian + SYNC xuống DB (persist). Lỗi DB không được
        làm chết worker → nuốt lỗi, chỉ log."""
        self.updated_at = _now()
        try:
            jobs_repo.save(self)
        except Exception:
            logger.exception("sync job %s xuống DB lỗi (bỏ qua)", self.id[:8])


def _row_to_job(row: dict) -> Job:
    """DB row → Job object (dùng khi đọc job cũ sau restart; không có RAM state)."""
    j = Job(id=row["id"], text=row["text"], voice=row["voice_id"], style=row["style"],
            temperature=row["temperature"], max_chars=row["max_chars"],
            user_ref=row["user_ref"], mode=row["mode"], status=row["status"],
            total_chunks=row["total_chunks"], done_chunks=row["done_chunks"],
            progress=row["progress"], audio_url=row["audio_url"], audio_key=row["audio_key"],
            duration_sec=row["duration_sec"], elapsed_sec=row["elapsed_sec"],
            sample_rate=row["sample_rate"], instance_id=row["instance_id"],
            error=row["error"])
    j.created_at = row["created_at"]
    j.updated_at = row["updated_at"]
    return j


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, text: str, voice: Optional[str], style: str, temperature: float,
               max_chars: int, mode: str = MODE_CPU, voice_record: Optional[dict] = None,
               user_ref: str = "default") -> Job:
        job = Job(id=str(uuid.uuid4()), text=text, voice=voice,
                  style=style or DEFAULT_STYLE, temperature=temperature,
                  max_chars=max_chars, mode=mode, voice_record=voice_record,
                  user_ref=user_ref)
        with self._lock:
            self._jobs[job.id] = job
        job.touch()             # ghi bản ghi đầu tiên xuống DB
        threading.Thread(target=self._run, args=(job,), daemon=True,
                         name=f"job-{job.id[:8]}").start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        """RAM trước (job đang chạy, tiến độ realtime); else DB (job cũ)."""
        job = self._jobs.get(job_id)
        if job is not None:
            return job
        row = jobs_repo.get(job_id)
        return _row_to_job(row) if row else None

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.status in ACTIVE_STATUSES:
            job.cancel.set()
            return True
        return False

    def active_count(self) -> int:
        try:
            return jobs_repo.active_count()
        except Exception:
            return sum(1 for j in self._jobs.values() if j.status in ACTIVE_STATUSES)

    # ── Worker: rẽ nhánh theo mode ───────────────────────────────────────────
    def _run(self, job: Job) -> None:
        try:
            if job.mode == "gpu":
                from . import gpu_vastai
                gpu_vastai.run(job)
            else:
                from . import cpu_onnx
                cpu_onnx.run(job)
        except Exception as e:            # backstop — mọi lỗi không lường trước
            logger.exception("job %s thất bại", job.id[:8])
            job.status = ERROR
            job.error = f"{type(e).__name__}: {e}"
            job.touch()


manager = JobManager()
