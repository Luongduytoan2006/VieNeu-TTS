"""Async TTS job orchestration (durable).

The BE owns the durable job (Postgres); the model-server runs the actual synthesis.
On ``start`` a worker thread submits the job to the model-server, then polls it and
mirrors progress % into the DB until it finishes. A per-job cancel Event propagates
cancellation to the model-server. The finished audio lives on the model-server's
storage (R2); we store its URL.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

from ..config import settings
from ..models import (
    STATUS_CANCELLED, STATUS_DONE, STATUS_ERROR, STATUS_RUNNING, Job,
)
from ..repositories import job_repository as repo
from .model_client import ModelClientError, client

logger = logging.getLogger("Vieneu.VPS.job_service")


class JobManager:
    def __init__(self) -> None:
        self._threads: Dict[str, threading.Thread] = {}
        self._cancel: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────────────────────
    def create(self, text: str, voice: Optional[str], style: str, temperature: float) -> str:
        """Persist a queued job and spawn its worker; return the job id."""
        job_id = repo.create(text, voice, style, temperature)
        self._start(job_id)
        return job_id

    def _start(self, job_id: str) -> None:
        ev = threading.Event()
        with self._lock:
            self._cancel[job_id] = ev
            t = threading.Thread(target=self._run, args=(job_id, ev), daemon=True,
                                 name=f"tts-{job_id[:8]}")
            self._threads[job_id] = t
        t.start()

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            ev = self._cancel.get(job_id)
        if ev is not None:
            ev.set()
            return True
        return False

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._threads.values() if t.is_alive())

    # ── Worker ───────────────────────────────────────────────────────────────
    def _run(self, job_id: str, cancel: threading.Event) -> None:
        try:
            self._run_remote(job_id, cancel)
        finally:
            with self._lock:
                self._threads.pop(job_id, None)
                self._cancel.pop(job_id, None)

    def _run_remote(self, job_id: str, cancel: threading.Event) -> None:
        job = repo.get(job_id)
        if job is None:
            return
        text, voice, style, temperature = job.text, job.voice, job.style, job.temperature
        max_chars = settings.MAX_CHARS_PER_CHUNK

        # Submit to the model-server.
        try:
            rj = client.create_job(text, voice, style, temperature, max_chars)
        except ModelClientError as e:
            self._fail(job_id, f"model-server: {e.detail}")
            return
        except Exception as e:
            self._fail(job_id, f"model-server unreachable: {e}")
            return

        remote_id = rj["id"]
        repo.update(job_id, status=STATUS_RUNNING, remote_job_id=remote_id)

        # Mirror the remote job's progress into our DB; propagate cancellation.
        while True:
            if cancel.is_set():
                try:
                    client.cancel_job(remote_id)
                except Exception:
                    pass
                repo.update(job_id, status=STATUS_CANCELLED)
                return
            try:
                d = client.get_job(remote_id)
            except Exception:
                time.sleep(settings.POLL_INTERVAL_SEC)
                continue

            repo.update(job_id, total_chunks=d.get("total_chunks", 0),
                        done_chunks=d.get("done_chunks", 0), progress=d.get("progress", 0.0))
            st = d.get("status")
            if st == "done":
                repo.update(job_id, status=STATUS_DONE, progress=100.0,
                            audio_url=d.get("audio_url"), duration_sec=d.get("duration_sec"),
                            elapsed_sec=d.get("elapsed_sec"), sample_rate=d.get("sample_rate"))
                logger.info("✅ remote job %s done → %s", job_id[:8], d.get("audio_url"))
                return
            if st == "cancelled":
                repo.update(job_id, status=STATUS_CANCELLED)
                return
            if st == "error":
                self._fail(job_id, d.get("error") or "model-server error")
                return
            time.sleep(settings.POLL_INTERVAL_SEC)

    def _fail(self, job_id: str, msg: str) -> None:
        repo.update(job_id, status=STATUS_ERROR, error=msg)


manager = JobManager()


def recover_stale_jobs() -> int:
    """On startup, mark any job left queued/running (process died) as error."""
    return repo.mark_stale_as_error()


# ── Result access ────────────────────────────────────────────────────────────
def get_job(job_id: str) -> Optional[Job]:
    return repo.get(job_id)


def fetch_audio(url: str) -> bytes:
    return client.fetch_audio(url)
