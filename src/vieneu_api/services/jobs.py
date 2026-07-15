"""Async TTS job manager.

Each job runs in its own worker thread. The text is split into chunks (the unit
of work); the worker synthesizes them one by one, updating ``done_chunks`` /
``progress`` in the DB after each, and checking a per-job cancel Event BETWEEN
chunks so a client can stop mid-run. When all chunks are done the waveforms are
joined and written to ``AUDIO_DIR/{job_id}.wav``; the DB row flips to ``done``
with the download path.

Progress granularity is one chunk (matches how the engine actually works — a
chunk's own generation loop is not interruptible). % = done_chunks/total*100.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict

import numpy as np
import soundfile as sf

from ..config import settings
from ..db import session_scope
from .. import models
from ..models import (
    ACTIVE_STATUSES, STATUS_CANCELLED, STATUS_DONE, STATUS_ERROR, STATUS_RUNNING, Job,
)
from .tts_service import service

logger = logging.getLogger("Vieneu.API.jobs")


class JobManager:
    def __init__(self) -> None:
        self._threads: Dict[str, threading.Thread] = {}
        self._cancel: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────────────────────
    def start(self, job_id: str) -> None:
        """Spawn the worker thread for an already-persisted queued job."""
        ev = threading.Event()
        with self._lock:
            self._cancel[job_id] = ev
            t = threading.Thread(target=self._run, args=(job_id, ev), daemon=True, name=f"tts-{job_id[:8]}")
            self._threads[job_id] = t
        t.start()

    def cancel(self, job_id: str) -> bool:
        """Signal a running job to stop. Returns True if it was active."""
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
            if settings.MODE == "remote":
                self._run_remote(job_id, cancel)
            else:
                self._run_local(job_id, cancel)
        finally:
            with self._lock:
                self._threads.pop(job_id, None)
                self._cancel.pop(job_id, None)

    # ── Remote worker (PA3): submit to the GPU model-server, mirror progress ──
    def _run_remote(self, job_id: str, cancel: threading.Event) -> None:
        from .model_client import client, ModelClientError
        try:
            with session_scope() as s:
                job = s.get(Job, job_id)
                if job is None:
                    return
                text, voice, style, temperature, max_chars = (
                    job.text, job.voice, job.style, job.temperature, settings.MAX_CHARS_PER_CHUNK)

            # Submit to the model-server.
            try:
                rj = client.create_job(text, voice, style, temperature, max_chars)
            except ModelClientError as e:
                self._fail(job_id, f"model-server: {e.detail}")
                return
            remote_id = rj["id"]
            self._update(job_id, status=STATUS_RUNNING, remote_job_id=remote_id)

            # Mirror the remote job's progress into our DB; propagate cancellation.
            while True:
                if cancel.is_set():
                    try:
                        client.cancel_job(remote_id)
                    except Exception:
                        pass
                    self._update(job_id, status=STATUS_CANCELLED)
                    return
                try:
                    d = client.get_job(remote_id)
                except Exception:
                    time.sleep(1.5)
                    continue
                self._update(job_id, total_chunks=d.get("total_chunks", 0),
                             done_chunks=d.get("done_chunks", 0), progress=d.get("progress", 0.0))
                st = d.get("status")
                if st == "done":
                    self._update(job_id, status=STATUS_DONE, progress=100.0,
                                 audio_url=d.get("audio_url"), duration_sec=d.get("duration_sec"),
                                 elapsed_sec=d.get("elapsed_sec"), sample_rate=d.get("sample_rate"))
                    logger.info("✅ remote job %s done → %s", job_id[:8], d.get("audio_url"))
                    return
                if st == "cancelled":
                    self._update(job_id, status=STATUS_CANCELLED)
                    return
                if st == "error":
                    self._fail(job_id, d.get("error") or "model-server error")
                    return
                time.sleep(1.2)
        except Exception as e:
            logger.exception("remote job %s failed", job_id[:8])
            self._fail(job_id, f"{type(e).__name__}: {e}")

    # ── Local worker (PA1): model in-process ─────────────────────────────────
    def _run_local(self, job_id: str, cancel: threading.Event) -> None:
        t0 = time.time()
        try:
            # Load the job spec.
            with session_scope() as s:
                job = s.get(Job, job_id)
                if job is None:
                    return
                text, voice, style, temperature, max_chars = (
                    job.text, job.voice, job.style, job.temperature,
                    settings.MAX_CHARS_PER_CHUNK,
                )

            chunks, gaps = service.split_chunks(text, max_chars)
            total = len(chunks)
            if total == 0:
                self._fail(job_id, "Văn bản rỗng sau chuẩn hóa.")
                return

            self._update(job_id, status=STATUS_RUNNING, total_chunks=total, done_chunks=0, progress=0.0)

            wavs = []
            for i, chunk in enumerate(chunks):
                if cancel.is_set():
                    self._update(job_id, status=STATUS_CANCELLED)
                    logger.info("⏹️ job %s cancelled at chunk %d/%d", job_id[:8], i, total)
                    return
                wav = service.synth_chunk(chunk, voice, style, temperature)
                if wav is not None and len(wav) > 0:
                    wavs.append(wav)
                done = i + 1
                self._update(job_id, done_chunks=done, progress=round(done / total * 100.0, 2))

            # One more cancel check before the (cheap) join/write.
            if cancel.is_set():
                self._update(job_id, status=STATUS_CANCELLED)
                return

            if not wavs:
                self._fail(job_id, "Không sinh được audio nào.")
                return

            from vieneu_utils.core_utils import join_audio_chunks, gaps_to_silence
            sr = int(service.sample_rate or 48000)
            final = join_audio_chunks(wavs, sr=sr, silence_ps=gaps_to_silence(gaps))
            out_path = settings.AUDIO_DIR / f"{job_id}.wav"
            sf.write(str(out_path), final, sr)

            duration = len(final) / sr if sr else 0.0
            self._update(
                job_id, status=STATUS_DONE, progress=100.0,
                audio_path=str(out_path), duration_sec=round(duration, 3),
                elapsed_sec=round(time.time() - t0, 3), sample_rate=sr,
            )
            logger.info("✅ job %s done: %.1fs audio in %.1fs", job_id[:8], duration, time.time() - t0)
        except Exception as e:
            logger.exception("job %s failed", job_id[:8])
            self._fail(job_id, f"{type(e).__name__}: {e}")

    # ── DB helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _update(job_id: str, **fields) -> None:
        with session_scope() as s:
            job = s.get(Job, job_id)
            if job is None:
                return
            for k, v in fields.items():
                setattr(job, k, v)

    def _fail(self, job_id: str, msg: str) -> None:
        self._update(job_id, status=STATUS_ERROR, error=msg)


manager = JobManager()


def recover_stale_jobs() -> int:
    """On startup, mark any job left 'queued'/'running' (process died) as error."""
    n = 0
    with session_scope() as s:
        for job in s.query(Job).filter(Job.status.in_(list(ACTIVE_STATUSES))).all():
            job.status = STATUS_ERROR
            job.error = "Server restarted while job was in progress."
            n += 1
    return n
