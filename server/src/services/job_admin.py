"""job_admin — API quản lý lịch sử TTS jobs.

Đọc/đổi tên/xóa job đã persist trong PostgreSQL. Xóa job cũng xóa audio trong
storage nếu còn file. Không dùng cho job đang queued/running.
"""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Iterable, Optional

from ..repositories import jobs_repo
from ..schemas import JobDeleteResponse, JobInfo, JobsResponse, JobsSummary
from ..storage import get_storage
from .jobs import ACTIVE_STATUSES, manager


class JobAdminError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _default_name(job_id: str, audio_key: Optional[str]) -> str:
    if audio_key:
        return PurePosixPath(audio_key).name[:50]
    return f"{job_id}.wav"


def _download_url(job_id: str, status: str) -> Optional[str]:
    return f"/api/v1/tts/{job_id}/download" if status == "done" else None


def _to_info(row: dict) -> JobInfo:
    name_audio = row.get("name_audio") or _default_name(row["id"], row.get("audio_key"))
    return JobInfo(
        id=row["id"], user_ref=row["user_ref"], text=row["text"],
        voice=row.get("voice_id"), style=row["style"],
        temperature=row["temperature"], max_chars=row["max_chars"],
        mode=row["mode"], status=row["status"], progress=row["progress"],
        done_chunks=row["done_chunks"], total_chunks=row["total_chunks"],
        name_audio=name_audio, audio_key=row.get("audio_key"),
        audio_size_bytes=row.get("audio_size_bytes"),
        duration_sec=row.get("duration_sec"), elapsed_sec=row.get("elapsed_sec"),
        sample_rate=row.get("sample_rate"), instance_id=row.get("instance_id"),
        error=row.get("error"), download_url=_download_url(row["id"], row["status"]),
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


def _summary(items: Iterable[JobInfo]) -> JobsSummary:
    s = JobsSummary()
    for job in items:
        s.total_jobs += 1
        if job.status == "queued":
            s.queued_jobs += 1
        elif job.status == "running":
            s.running_jobs += 1
        elif job.status == "done":
            s.done_jobs += 1
        elif job.status == "cancelled":
            s.cancelled_jobs += 1
        elif job.status == "error":
            s.error_jobs += 1
        if job.mode == "cpu":
            s.cpu_jobs += 1
        elif job.mode == "gpu":
            s.gpu_jobs += 1
        s.total_duration_sec += float(job.duration_sec or 0)
        s.total_elapsed_sec += float(job.elapsed_sec or 0)
        s.total_audio_size_bytes += int(job.audio_size_bytes or 0)
    s.total_duration_sec = round(s.total_duration_sec, 3)
    s.total_elapsed_sec = round(s.total_elapsed_sec, 3)
    return s


def _response(rows: list[dict]) -> JobsResponse:
    jobs = [_to_info(r) for r in rows]
    return JobsResponse(summary=_summary(jobs), jobs=jobs)


def list_all() -> JobsResponse:
    return _response(jobs_repo.list_all())


def list_user(user_ref: str) -> JobsResponse:
    return _response(jobs_repo.list_by_user(user_ref))


def get(job_id: str) -> JobInfo:
    row = jobs_repo.get(job_id)
    if row is None:
        raise JobAdminError(404, f"Job '{job_id}' không tồn tại.")
    return _to_info(row)


def update_name(job_id: str, name_audio: str) -> JobInfo:
    clean_name = name_audio.strip()
    row = jobs_repo.update_name(job_id, clean_name)
    if row is None:
        raise JobAdminError(404, f"Job '{job_id}' không tồn tại.")
    manager.rename(job_id, clean_name)
    return _to_info(row)


def delete(job_id: str) -> JobDeleteResponse:
    row = jobs_repo.get(job_id)
    if row is None:
        raise JobAdminError(404, f"Job '{job_id}' không tồn tại.")
    if row["status"] in ACTIVE_STATUSES:
        raise JobAdminError(409, f"Job đang {row['status']}; hủy hoặc chờ xong trước khi xóa.")

    audio_key = row.get("audio_key")
    deleted_audio = False
    if audio_key:
        st = get_storage()
        if st.exists(audio_key):
            st.delete(audio_key)
            deleted_audio = True

    deleted = jobs_repo.delete(job_id)
    manager.forget(job_id)
    return JobDeleteResponse(
        id=job_id, deleted_job=deleted is not None,
        deleted_audio=deleted_audio, audio_key=audio_key,
    )
