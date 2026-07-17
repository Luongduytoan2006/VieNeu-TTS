"""Quản lý job TTS async (in-memory) — vì chỉ 1 server nên không cần DB.

Job có uuid, chạy trong 1 thread riêng, stream % tiến độ. Việc synth thật được
đẩy xuống backend theo mode: ``cpu_onnx`` (in-process) hoặc ``gpu_vastai`` (Vast.ai).
Có cancel Event để hủy giữa chừng.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

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
    mode: str = MODE_CPU
    # Record giọng (dict: speaker_emb+codes numpy) đã lấy từ DB lúc tạo job.
    # Worker cpu/gpu dùng thẳng — không tra lại catalog. Không serialize (numpy).
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
    error: Optional[str] = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    cancel: threading.Event = field(default_factory=threading.Event)

    def touch(self) -> None:
        self.updated_at = _now()


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, text: str, voice: Optional[str], style: str, temperature: float,
               max_chars: int, mode: str = MODE_CPU,
               voice_record: Optional[dict] = None) -> Job:
        job = Job(id=str(uuid.uuid4()), text=text, voice=voice,
                  style=style or DEFAULT_STYLE, temperature=temperature,
                  max_chars=max_chars, mode=mode, voice_record=voice_record)
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True,
                         name=f"job-{job.id[:8]}").start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.status in ACTIVE_STATUSES:
            job.cancel.set()
            return True
        return False

    def active_count(self) -> int:
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
