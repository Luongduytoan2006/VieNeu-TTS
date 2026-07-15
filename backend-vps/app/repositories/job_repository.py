"""Data-access for Job rows. All ORM access for jobs lives here."""
from __future__ import annotations

from typing import Optional

from ..database import session_scope
from ..models import ACTIVE_STATUSES, STATUS_ERROR, Job


def create(text: str, voice: Optional[str], style: str, temperature: float) -> str:
    """Insert a queued job; return its id."""
    with session_scope() as s:
        job = Job(text=text, voice=voice, style=style, temperature=temperature)
        s.add(job)
        s.flush()
        return job.id


def get(job_id: str) -> Optional[Job]:
    """Return a detached Job (safe to read after the session closes)."""
    with session_scope() as s:
        job = s.get(Job, job_id)
        if job is not None:
            s.expunge(job)
        return job


def update(job_id: str, **fields) -> None:
    with session_scope() as s:
        job = s.get(Job, job_id)
        if job is None:
            return
        for k, v in fields.items():
            setattr(job, k, v)


def mark_stale_as_error() -> int:
    """On startup: any job still queued/running (process died) → error."""
    n = 0
    with session_scope() as s:
        for job in s.query(Job).filter(Job.status.in_(list(ACTIVE_STATUSES))).all():
            job.status = STATUS_ERROR
            job.error = "Server restarted while job was in progress."
            n += 1
    return n
