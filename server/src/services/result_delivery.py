from __future__ import annotations

from ..storage import get_storage
from . import delivery


def store_result(job, data: bytes, *, duration_sec: float,
                 elapsed_sec: float, sample_rate: int) -> None:
    if job.delivery:
        result = delivery.deliver(
            job,
            data,
            duration_sec=duration_sec,
            elapsed_sec=elapsed_sec,
            sample_rate=sample_rate,
        )
        job.audio_key = None
        job.audio_url = None
        job.audio_size_bytes = result.size
        job.delivery_unconfirmed = not result.confirmed
        return

    key = f"audio/{job.id}.wav"
    job.audio_url = get_storage().put(key, data, "audio/wav")
    job.audio_key = key
    job.audio_size_bytes = len(data)
