"""test_jobs_api — API quản lý jobs, OFFLINE, KHÔNG model/R2/GPU."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_SERVER = Path(__file__).resolve().parents[1] / "server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))

os.environ["MODEL_EAGER_LOAD"] = "0"
os.environ["STORAGE_BACKEND"] = "local"
os.environ["ACCESS_SECRET_KEY"] = ""

import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from src.config import settings  # noqa: E402
from src.controllers import tts as tts_controller  # noqa: E402
from src.repositories import jobs_repo  # noqa: E402
from src.schemas import DeliveryCapability  # noqa: E402
from src.services import create_audio, job_admin  # noqa: E402
from src.services.jobs import Job  # noqa: E402

settings.API_KEY = ""

NOW = datetime(2026, 8, 10, tzinfo=timezone.utc)


def _row(job_id: str = "job-1", *, user_ref: str = "u1", mode: str = "cpu",
         status: str = "done", size: int | None = 123, audio_key: str | None = None) -> dict:
    return {
        "id": job_id, "user_ref": user_ref, "text": "xin chao", "voice_id": "Duy Toàn",
        "style": "tu_nhien", "temperature": 0.8, "max_chars": 256,
        "mode": mode, "status": status, "progress": 100.0,
        "done_chunks": 1, "total_chunks": 1,
        "audio_key": audio_key or f"audio/{job_id}.wav",
        "audio_url": "http://example.test/file.wav", "name_audio": f"{job_id}.wav",
        "audio_size_bytes": size, "duration_sec": 2.5, "elapsed_sec": 1.2,
        "sample_rate": 48000, "instance_id": None, "error": None,
        "created_at": NOW, "updated_at": NOW,
    }


def _client() -> TestClient:
    return TestClient(main.app)


def test_list_all_jobs_has_summary():
    rows = [
        _row("cpu-1", mode="cpu", status="done", size=100),
        _row("gpu-1", mode="gpu", status="error", size=None),
    ]
    with mock.patch.object(jobs_repo, "list_all", return_value=rows):
        r = _client().get("/api/v1/jobs")
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["total_jobs"] == 2
    assert body["summary"]["cpu_jobs"] == 1
    assert body["summary"]["gpu_jobs"] == 1
    assert body["summary"]["done_jobs"] == 1
    assert body["summary"]["error_jobs"] == 1
    assert body["summary"]["total_audio_size_bytes"] == 100
    assert len(body["jobs"]) == 2


def test_list_user_jobs_empty_ok():
    with mock.patch.object(jobs_repo, "list_by_user", return_value=[]) as list_by_user:
        r = _client().get("/api/v1/jobs/users/no-data")
    assert r.status_code == 200
    assert r.json()["summary"]["total_jobs"] == 0
    assert r.json()["jobs"] == []
    list_by_user.assert_called_once_with("no-data")


def test_get_job_detail():
    with mock.patch.object(jobs_repo, "get", return_value=_row("job-detail")):
        r = _client().get("/api/v1/jobs/job-detail")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "job-detail"
    assert body["audio_key"] == "audio/job-detail.wav"
    assert body["audio_size_bytes"] == 123


def test_update_job_name_audio():
    updated = _row("job-update")
    updated["name_audio"] = "new-name.wav"
    with mock.patch.object(jobs_repo, "update_name", return_value=updated) as update_name:
        r = _client().patch("/api/v1/jobs/job-update", json={"name_audio": "new-name.wav"})
    assert r.status_code == 200
    assert r.json()["name_audio"] == "new-name.wav"
    update_name.assert_called_once_with("job-update", "new-name.wav")


def test_update_job_name_too_long_422():
    r = _client().patch("/api/v1/jobs/job-update", json={"name_audio": "x" * 51})
    assert r.status_code == 422


class _Storage:
    def __init__(self):
        self.deleted = []

    def exists(self, key):
        return True

    def delete(self, key):
        self.deleted.append(key)


def test_delete_job_deletes_db_and_storage():
    st = _Storage()
    row = _row("job-delete")
    with mock.patch.object(jobs_repo, "get", return_value=row), \
            mock.patch.object(jobs_repo, "delete", return_value=row) as delete_row, \
            mock.patch.object(job_admin, "get_storage", return_value=st), \
            mock.patch.object(job_admin.manager, "forget") as forget:
        r = _client().delete("/api/v1/jobs/job-delete")
    assert r.status_code == 200
    assert r.json()["deleted_job"] is True
    assert r.json()["deleted_audio"] is True
    assert st.deleted == ["audio/job-delete.wav"]
    delete_row.assert_called_once_with("job-delete")
    forget.assert_called_once_with("job-delete")


def test_delete_running_job_blocked_409():
    with mock.patch.object(jobs_repo, "get", return_value=_row("job-run", status="running")):
        r = _client().delete("/api/v1/jobs/job-run")
    assert r.status_code == 409


def test_create_tts_returns_name_and_timestamps():
    fake_job = SimpleNamespace(
        id="created-1", status="queued", mode="cpu", name_audio="custom.wav",
        audio_key=None, audio_size_bytes=None, created_at=NOW, updated_at=NOW,
    )
    with mock.patch.object(create_audio, "create", return_value=fake_job) as create:
        r = _client().post("/api/v1/tts", json={
            "text": "xin chao", "mode": "cpu", "name_audio": "custom.wav",
        })
    assert r.status_code == 202
    body = r.json()
    assert body["name_audio"] == "custom.wav"
    assert "created_at" in body
    assert "updated_at" in body
    assert create.call_args.kwargs["name_audio"] == "custom.wav"


def test_create_tts_threads_external_delivery_without_exposing_capability():
    generation_id = "4f8efef0-bf4f-41e7-a193-b97d51d15d90"
    fake_job = SimpleNamespace(
        id="created-2", status="queued", mode="cpu", name_audio="custom.wav",
        audio_key=None, audio_size_bytes=None, created_at=NOW, updated_at=NOW,
        delivery_kind="openvoice", external_generation_id=generation_id,
    )
    delivery = {"generation_id": generation_id, "capability_token": "x" * 43}
    with mock.patch.object(create_audio, "create", return_value=fake_job) as create:
        r = _client().post("/api/v1/tts", json={
            "text": "xin chao", "mode": "cpu", "delivery": delivery,
        })

    assert r.status_code == 202
    assert r.json()["download_url"] is None
    assert "delivery" not in r.json()
    assert create.call_args.kwargs["delivery"] == DeliveryCapability.model_validate(delivery)


def test_lookup_by_generation_is_owner_scoped():
    generation_id = "4f8efef0-bf4f-41e7-a193-b97d51d15d90"
    job = Job(
        id="job-lookup", text="xin chao", voice=None, style="tu_nhien",
        temperature=0.8, max_chars=256, user_ref="u1", status="queued",
        delivery_kind="openvoice", external_generation_id=generation_id,
    )
    with mock.patch.object(
        tts_controller.manager,
        "get_by_generation",
        side_effect=lambda user, _generation: job if user == "u1" else None,
    ) as lookup:
        found = _client().get(
            f"/api/v1/tts/by-generation/{generation_id}", headers={"X-User-Id": "u1"},
        )
        hidden = _client().get(
            f"/api/v1/tts/by-generation/{generation_id}", headers={"X-User-Id": "u2"},
        )

    assert found.status_code == 200
    assert found.json()["id"] == "job-lookup"
    assert hidden.status_code == 404
    assert lookup.call_args_list == [mock.call("u1", generation_id), mock.call("u2", generation_id)]


def test_external_delivery_download_is_gone_even_after_ram_capability_is_lost():
    job = Job(
        id="job-external", text="xin chao", voice=None, style="tu_nhien",
        temperature=0.8, max_chars=256, user_ref="u1", status="done",
        delivery=None, delivery_kind="openvoice",
        external_generation_id="4f8efef0-bf4f-41e7-a193-b97d51d15d90",
    )
    with mock.patch.object(tts_controller.manager, "get", return_value=job):
        response = _client().get(
            "/api/v1/tts/job-external/download", headers={"X-User-Id": "u1"},
        )

    assert response.status_code == 410


def _run() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{'ALL PASSED' if not failures else str(failures) + ' FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run())
