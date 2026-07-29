"""CPU backend — synth THẲNG trên máy này (in-process, ONNX/CPU hoặc CUDA nếu có).

Model đã nạp sẵn qua ``src.engine``. Cắt chunk → synth từng chunk (cập nhật % vào
job) → ghép → ghi WAV → đẩy lên storage (local ``/files`` hoặc R2). Không HTTP,
không provider ngoài — đây là đường mặc định của server.
"""
from __future__ import annotations

import io
import logging
import time
import wave

import numpy as np

from ..engine import engine
from ..storage import get_storage
from .jobs import CANCELLED, DONE, ERROR, RUNNING, Job

logger = logging.getLogger("Vieneu.CPU")


def _wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def run(job: Job) -> None:
    """Chạy toàn bộ 1 job TTS trên CPU (blocking, gọi trong thread của job)."""
    t0 = time.time()

    # Đảm bảo model đã sẵn sàng (lazy-load nếu chưa eager).
    if not engine.loaded:
        engine.load()

    chunks, gaps = engine.split_chunks(job.text, job.max_chars)
    job.total_chunks = len(chunks)
    if job.total_chunks == 0:
        job.status = ERROR
        job.error = "Văn bản rỗng."
        job.touch()
        return

    job.status = RUNNING
    job.touch()

    wavs = []
    for i, chunk in enumerate(chunks):
        if job.cancel.is_set():
            job.status = CANCELLED
            job.touch()
            logger.info("⏹️ job %s cancelled %d/%d", job.id[:8], i, job.total_chunks)
            return
        wav = engine.synth_chunk(chunk, job.voice_record, job.style, job.temperature)
        if wav is not None and len(wav) > 0:
            wavs.append(wav)
        job.done_chunks = i + 1
        job.progress = round((i + 1) / job.total_chunks * 100.0, 2)
        job.touch()

    if job.cancel.is_set():
        job.status = CANCELLED
        job.touch()
        return
    if not wavs:
        job.status = ERROR
        job.error = "Không sinh được audio."
        job.touch()
        return

    from vieneu_utils.core_utils import gaps_to_silence, join_audio_chunks
    sr = engine.sample_rate
    final = join_audio_chunks(wavs, sr=sr, silence_ps=gaps_to_silence(gaps))
    data = _wav_bytes(final, sr)

    key = f"audio/{job.id}.wav"
    url = get_storage().put(key, data, "audio/wav")

    job.audio_key = key
    job.audio_url = url
    job.duration_sec = round(len(final) / sr, 3)
    job.elapsed_sec = round(time.time() - t0, 3)
    job.sample_rate = sr
    job.status = DONE
    job.progress = 100.0
    job.touch()
    logger.info("✅ CPU job %s xong: %.1fs → %s", job.id[:8], job.duration_sec, url)
