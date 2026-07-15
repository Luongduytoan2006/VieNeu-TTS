"""In-memory job manager for the model-server.

The BE holds the durable (Postgres) job; this side keeps a lightweight in-memory
job while synthesis runs, streams progress %, and on completion writes the WAV to
storage and exposes its URL. No DB here — the BE is the source of truth.
"""
from __future__ import annotations

import io
import logging
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .engine import engine
from .storage import make_storage

logger = logging.getLogger("Vieneu.ModelServer.jobs")

STORAGE = make_storage()

QUEUED, RUNNING, DONE, CANCELLED, ERROR = "queued", "running", "done", "cancelled", "error"


@dataclass
class Job:
    id: str
    text: str
    voice: Optional[str]
    style: str
    temperature: float
    max_chars: int
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
    cancel: threading.Event = field(default_factory=threading.Event)


def _wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
    return buf.getvalue()


class ModelJobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, text, voice, style, temperature, max_chars) -> Job:
        job = Job(id=str(uuid.uuid4()), text=text, voice=voice, style=style,
                  temperature=temperature, max_chars=max_chars)
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True,
                         name=f"model-{job.id[:8]}").start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.status in (QUEUED, RUNNING):
            job.cancel.set()
            return True
        return False

    def active_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status in (QUEUED, RUNNING))

    def _run(self, job: Job) -> None:
        t0 = time.time()
        try:
            chunks, gaps = engine.split_chunks(job.text, job.max_chars)
            job.total_chunks = len(chunks)
            if job.total_chunks == 0:
                job.status = ERROR; job.error = "Văn bản rỗng."; return
            job.status = RUNNING
            wavs = []
            for i, chunk in enumerate(chunks):
                if job.cancel.is_set():
                    job.status = CANCELLED
                    logger.info("⏹️ model job %s cancelled %d/%d", job.id[:8], i, job.total_chunks)
                    return
                wav = engine.synth_chunk(chunk, job.voice, job.style, job.temperature)
                if wav is not None and len(wav) > 0:
                    wavs.append(wav)
                job.done_chunks = i + 1
                job.progress = round((i + 1) / job.total_chunks * 100.0, 2)
            if job.cancel.is_set():
                job.status = CANCELLED; return
            if not wavs:
                job.status = ERROR; job.error = "Không sinh được audio."; return

            from vieneu_utils.core_utils import join_audio_chunks, gaps_to_silence
            sr = engine.sample_rate
            final = join_audio_chunks(wavs, sr=sr, silence_ps=gaps_to_silence(gaps))
            data = _wav_bytes(final, sr)
            key = f"audio/{job.id}.wav"
            url = STORAGE.put(key, data, "audio/wav")   # ← GPU-server lưu THẲNG vào storage
            job.audio_key = key
            job.audio_url = url
            job.duration_sec = round(len(final) / sr, 3)
            job.elapsed_sec = round(time.time() - t0, 3)
            job.sample_rate = sr
            job.status = DONE
            logger.info("✅ model job %s done: %.1fs → %s", job.id[:8], job.duration_sec, url)
        except Exception as e:
            logger.exception("model job %s failed", job.id[:8])
            job.status = ERROR; job.error = f"{type(e).__name__}: {e}"


manager = ModelJobManager()
