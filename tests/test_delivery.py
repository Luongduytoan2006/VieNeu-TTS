from __future__ import annotations

import base64
import hashlib
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

import numpy as np
import pytest

_SERVER = Path(__file__).resolve().parents[1] / "server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))

from src.config import settings  # noqa: E402
from src.repositories import jobs_repo  # noqa: E402
from src.schemas import DeliveryCapability  # noqa: E402
from src.services import cpu_onnx, delivery, result_delivery  # noqa: E402
from src.services.jobs import Job, JobManager  # noqa: E402

from src import db  # noqa: E402


class Response:
    def __init__(self, status_code: int, payload: dict | None = None,
                 headers: dict[str, str] | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self) -> dict:
        return self._payload


class Session:
    def __init__(self, posts: list[Response], puts: list[Response | Exception]):
        self.posts = list(posts)
        self.puts = list(puts)
        self.post_calls: list[tuple] = []
        self.put_calls: list[tuple] = []

    def post(self, *args, **kwargs):
        self.post_calls.append((args, kwargs))
        return self.posts.pop(0)

    def put(self, *args, **kwargs):
        self.put_calls.append((args, kwargs))
        result = self.puts.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _job() -> SimpleNamespace:
    return SimpleNamespace(
        id=str(uuid4()),
        delivery=DeliveryCapability(
            generation_id=uuid4(),
            capability_token="secret-capability-" + "x" * 44,
        ),
    )


def _grant(url: str, wav: bytes) -> dict:
    digest = base64.b64encode(
        hashlib.md5(wav, usedforsecurity=False).digest(),
    ).decode("ascii")
    return {
        "method": "PUT",
        "url": url,
        "headers": {
            "Content-Type": "audio/wav",
            "Content-MD5": digest,
            "If-None-Match": "*",
        },
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
    }


def _settings():
    return mock.patch.multiple(
        settings,
        OPENVOICE_BASE_URL="http://localhost:3011",
        OPENVOICE_R2_UPLOAD_ORIGIN="https://account.r2.cloudflarestorage.com",
        OPENVOICE_ALLOW_HTTP=True,
        OPENVOICE_DELIVERY_TIMEOUT=10,
        OPENVOICE_CALLBACK_RETRIES=3,
        MAX_DELIVERY_BYTES=1024,
    )


def test_delivery_uses_raw_url_exact_headers_and_md5():
    wav = b"RIFF-wave"
    job = _job()
    raw_url = "https://account.r2.cloudflarestorage.com/bucket/a%2Fb.wav?b=2&a=1%2B2"
    session = Session(
        posts=[Response(200, _grant(raw_url, wav)), Response(200)],
        puts=[Response(200, headers={"ETag": '"etag"'})],
    )

    with _settings(), mock.patch.object(delivery, "_session", return_value=session):
        result = delivery.deliver(
            job, wav, duration_sec=1.2, elapsed_sec=2.3, sample_rate=48000,
        )

    assert result.size == len(wav)
    assert result.confirmed is True
    put_args, put_kwargs = session.put_calls[0]
    assert put_args[0] == raw_url
    assert put_kwargs["headers"] == _grant(raw_url, wav)["headers"]
    assert put_kwargs["data"] == wav
    assert put_kwargs["allow_redirects"] is False
    grant_body = session.post_calls[0][1]["json"]
    assert grant_body["content_md5"] == _grant(raw_url, wav)["headers"]["Content-MD5"]
    assert session.post_calls[-1][1]["json"]["status"] == "ready"


def test_proxy_upload_adds_delivery_capability_authorization():
    wav = b"RIFF-wave"
    job = _job()
    raw_url = "http://localhost:3011/api/voice/internal/deliveries/generation/upload"
    session = Session(
        posts=[Response(200, _grant(raw_url, wav)), Response(200)],
        puts=[Response(201)],
    )

    with _settings(), mock.patch.object(delivery, "_session", return_value=session):
        delivery.deliver(
            job, wav, duration_sec=1.2, elapsed_sec=2.3, sample_rate=48000,
        )

    assert session.put_calls[0][1]["headers"]["Authorization"] == (
        f"Bearer {job.delivery.capability_token}"
    )


def test_invalid_grant_does_not_leak_signed_url_in_delivery_traceback():
    wav = b"RIFF"
    raw_url = "https://account.r2.cloudflarestorage.com/bucket/result.wav?secret=hidden\n"
    invalid = _grant(raw_url, wav)
    session = Session(posts=[Response(200, invalid), Response(200)], puts=[])

    with _settings(), mock.patch.object(delivery, "_session", return_value=session):
        try:
            delivery.deliver(
                _job(), wav, duration_sec=1, elapsed_sec=2, sample_rate=48000,
            )
        except delivery.DeliveryError as error:
            rendered = "".join(traceback.format_exception(error))
        else:
            raise AssertionError("invalid grant must fail")

    assert raw_url not in rendered


def test_delivery_retries_425_until_upstream_binding_exists():
    wav = b"RIFF"
    raw_url = "https://account.r2.cloudflarestorage.com/bucket/result.wav"
    session = Session(
        posts=[Response(425), Response(200, _grant(raw_url, wav)), Response(200)],
        puts=[Response(200)],
    )

    with _settings(), mock.patch.object(delivery, "_session", return_value=session), \
            mock.patch.object(delivery.time, "sleep") as sleep:
        result = delivery.deliver(
            _job(), wav, duration_sec=1, elapsed_sec=2, sample_rate=48000,
        )

    assert result.confirmed is True
    sleep.assert_called_once_with(1)


def test_delivery_retries_known_transient_upload_response():
    wav = b"RIFF"
    raw_url = "https://account.r2.cloudflarestorage.com/bucket/result.wav"
    session = Session(
        posts=[Response(200, _grant(raw_url, wav)), Response(200)],
        puts=[Response(503), Response(200)],
    )

    with _settings(), mock.patch.object(delivery, "_session", return_value=session), \
            mock.patch.object(delivery.time, "sleep") as sleep:
        result = delivery.deliver(
            _job(), wav, duration_sec=1, elapsed_sec=2, sample_rate=48000,
        )

    assert result.confirmed is True
    assert len(session.put_calls) == 2
    sleep.assert_called_once_with(1)


def test_delivery_requests_fresh_grant_after_expired_upload_response():
    wav = b"RIFF"
    first_url = "https://account.r2.cloudflarestorage.com/bucket/result.wav?grant=1"
    second_url = "https://account.r2.cloudflarestorage.com/bucket/result.wav?grant=2"
    session = Session(
        posts=[
            Response(200, _grant(first_url, wav)),
            Response(200, _grant(second_url, wav)),
            Response(200),
        ],
        puts=[Response(403), Response(200)],
    )

    with _settings(), mock.patch.object(delivery, "_session", return_value=session):
        result = delivery.deliver(
            _job(), wav, duration_sec=1, elapsed_sec=2, sample_rate=48000,
        )

    assert result.confirmed is True
    assert [call[0][0] for call in session.put_calls] == [first_url, second_url]


def test_put_response_loss_reconciles_ready_without_error_callback_or_overwrite():
    wav = b"RIFF"
    raw_url = "https://account.r2.cloudflarestorage.com/bucket/result.wav?secret=hidden"
    session = Session(
        posts=[Response(200, _grant(raw_url, wav)), Response(200)],
        puts=[delivery.requests.ConnectionError(f"lost response from {raw_url}")],
    )

    with _settings(), mock.patch.object(delivery, "_session", return_value=session):
        result = delivery.deliver(
            _job(), wav, duration_sec=1, elapsed_sec=2, sample_rate=48000,
        )

    assert result.confirmed is True
    assert len(session.put_calls) == 1
    callback_bodies = [call[1]["json"] for call in session.post_calls[1:]]
    assert [body["status"] for body in callback_bodies] == ["ready"]
    assert raw_url not in repr(result)


def test_missing_object_after_response_loss_gets_fresh_non_overwriting_grant():
    wav = b"RIFF"
    first_url = "https://account.r2.cloudflarestorage.com/bucket/result.wav?grant=old"
    second_url = "https://account.r2.cloudflarestorage.com/bucket/result.wav?grant=new"
    session = Session(
        posts=[
            Response(200, _grant(first_url, wav)),
            Response(409, {"statusMessage": "Delivered audio failed object verification"}),
            Response(200, _grant(second_url, wav)),
            Response(200),
        ],
        puts=[delivery.requests.ConnectionError("lost response"), Response(200)],
    )

    with _settings(), mock.patch.object(delivery, "_session", return_value=session):
        result = delivery.deliver(
            _job(), wav, duration_sec=1, elapsed_sec=2, sample_rate=48000,
        )

    assert result.confirmed is True
    assert [call[0][0] for call in session.put_calls] == [first_url, second_url]
    assert all(call[1]["headers"]["If-None-Match"] == "*" for call in session.put_calls)


def test_unrelated_completion_conflict_after_response_loss_does_not_retry_put():
    wav = b"RIFF"
    raw_url = "https://account.r2.cloudflarestorage.com/bucket/result.wav"
    session = Session(
        posts=[
            Response(200, _grant(raw_url, wav)),
            Response(409, {"statusMessage": "Voice generation job does not match"}),
        ],
        puts=[delivery.requests.ConnectionError("lost response")],
    )

    with _settings(), mock.patch.object(delivery, "_session", return_value=session):
        result = delivery.deliver(
            _job(), wav, duration_sec=1, elapsed_sec=2, sample_rate=48000,
        )

    assert result.confirmed is False
    assert len(session.put_calls) == 1


def test_unknown_reconciliation_after_response_loss_stays_unconfirmed_without_overwrite():
    wav = b"RIFF"
    raw_url = "https://account.r2.cloudflarestorage.com/bucket/result.wav?secret=hidden"
    session = Session(
        posts=[
            Response(200, _grant(raw_url, wav)),
            Response(503),
            Response(503),
            Response(503),
        ],
        puts=[delivery.requests.ConnectionError("lost response")],
    )

    with _settings(), mock.patch.object(delivery, "_session", return_value=session), \
            mock.patch.object(delivery.time, "sleep"):
        result = delivery.deliver(
            _job(), wav, duration_sec=1, elapsed_sec=2, sample_rate=48000,
        )

    assert result.confirmed is False
    assert len(session.put_calls) == 1
    callback_bodies = [call[1]["json"] for call in session.post_calls[1:]]
    assert {body["status"] for body in callback_bodies} == {"ready"}


def test_precondition_failed_reconciles_existing_object_without_overwrite():
    wav = b"RIFF"
    raw_url = "https://account.r2.cloudflarestorage.com/bucket/result.wav"
    session = Session(
        posts=[Response(200, _grant(raw_url, wav)), Response(200)],
        puts=[Response(412)],
    )

    with _settings(), mock.patch.object(delivery, "_session", return_value=session):
        result = delivery.deliver(
            _job(), wav, duration_sec=1, elapsed_sec=2, sample_rate=48000,
        )

    assert result.confirmed is True
    assert len(session.put_calls) == 1
    assert session.post_calls[-1][1]["json"]["status"] == "ready"


def test_callback_exhaustion_keeps_durable_upload_successful_but_unconfirmed():
    wav = b"RIFF"
    raw_url = "https://account.r2.cloudflarestorage.com/bucket/result.wav"
    session = Session(
        posts=[Response(200, _grant(raw_url, wav)), Response(503), Response(503), Response(503)],
        puts=[Response(200)],
    )

    with _settings(), mock.patch.object(delivery, "_session", return_value=session), \
            mock.patch.object(delivery.time, "sleep"):
        result = delivery.deliver(
            _job(), wav, duration_sec=1, elapsed_sec=2, sample_rate=48000,
        )

    assert result.confirmed is False


def test_grant_failure_sends_sanitized_error_completion():
    session = Session(posts=[Response(400), Response(200)], puts=[])

    with _settings(), mock.patch.object(delivery, "_session", return_value=session), \
            pytest.raises(delivery.DeliveryError, match=r"upload grant failed \(400\)"):
        delivery.deliver(
            _job(), b"RIFF", duration_sec=1, elapsed_sec=2, sample_rate=48000,
        )

    assert session.post_calls[-1][1]["json"]["status"] == "error"
    assert "http" not in session.post_calls[-1][1]["json"]["error"]


def test_expired_grant_is_replaced_before_upload():
    wav = b"RIFF"
    expired_url = "https://account.r2.cloudflarestorage.com/bucket/result.wav?grant=expired"
    fresh_url = "https://account.r2.cloudflarestorage.com/bucket/result.wav?grant=fresh"
    grant = _grant(expired_url, wav)
    grant["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    session = Session(
        posts=[Response(200, grant), Response(200, _grant(fresh_url, wav)), Response(200)],
        puts=[Response(200)],
    )

    with _settings(), mock.patch.object(delivery, "_session", return_value=session):
        result = delivery.deliver(
            _job(), wav, duration_sec=1, elapsed_sec=2, sample_rate=48000,
        )

    assert result.confirmed is True
    assert [call[0][0] for call in session.put_calls] == [fresh_url]


@pytest.mark.parametrize(("url", "allow_http", "allowed"), [
    ("http://localhost:3011/api/upload", True, True),
    ("http://host.docker.internal:3011/api/upload", True, True),
    ("http://localhost:3011/api/upload", False, False),
    ("https://127.0.0.1/upload", False, False),
    ("https://169.254.169.254/latest/meta-data", False, False),
    ("https://account.r2.cloudflarestorage.com:444/upload", False, False),
    ("https://evil.example/upload", False, False),
])
def test_upload_url_origin_and_private_address_policy(url: str, allow_http: bool, allowed: bool):
    base = "http://host.docker.internal:3011" if "host.docker.internal" in url else "http://localhost:3011"
    with mock.patch.multiple(
        settings,
        OPENVOICE_BASE_URL=base,
        OPENVOICE_R2_UPLOAD_ORIGIN="https://account.r2.cloudflarestorage.com",
        OPENVOICE_ALLOW_HTTP=allow_http,
    ):
        assert delivery._allowed_upload_url(url) is allowed


def test_result_delivery_bypasses_legacy_storage_for_external_job():
    job = SimpleNamespace(id=str(uuid4()), delivery=object(), audio_key="old", audio_url="old")
    delivered = delivery.DeliveryResult(size=4, etag=None, confirmed=False)

    with mock.patch.object(delivery, "deliver", return_value=delivered) as send, \
            mock.patch.object(result_delivery, "get_storage") as get_storage:
        result_delivery.store_result(
            job, b"RIFF", duration_sec=1, elapsed_sec=2, sample_rate=48000,
        )

    send.assert_called_once()
    get_storage.assert_not_called()
    assert job.audio_key is None
    assert job.audio_url is None
    assert job.audio_size_bytes == 4
    assert job.delivery_unconfirmed is True


def test_result_delivery_keeps_legacy_storage_path():
    job = SimpleNamespace(id="job-id", delivery=None)
    storage = mock.Mock()
    storage.put.return_value = "https://storage.example/audio/job-id.wav"

    with mock.patch.object(result_delivery, "get_storage", return_value=storage):
        result_delivery.store_result(
            job, b"RIFF", duration_sec=1, elapsed_sec=2, sample_rate=48000,
        )

    storage.put.assert_called_once_with("audio/job-id.wav", b"RIFF", "audio/wav")
    assert job.audio_key == "audio/job-id.wav"


def test_cpu_finalization_uses_shared_result_delivery():
    engine = SimpleNamespace(
        loaded=True,
        sample_rate=48000,
        split_chunks=mock.Mock(return_value=(["xin chao"], [])),
        synth_chunk=mock.Mock(return_value=np.zeros(48, dtype=np.float32)),
    )
    job = Job(
        id=str(uuid4()), text="xin chao", voice=None, style="tu_nhien",
        temperature=0.8, max_chars=256, delivery=DeliveryCapability(
            generation_id=uuid4(), capability_token="private-" + "x" * 43,
        ),
    )
    job.touch = mock.Mock()

    with mock.patch.object(cpu_onnx, "engine", engine), \
            mock.patch.object(cpu_onnx, "store_result") as store, \
            mock.patch("vieneu_utils.core_utils.gaps_to_silence", return_value=[]), \
            mock.patch(
                "vieneu_utils.core_utils.join_audio_chunks",
                return_value=np.zeros(48, dtype=np.float32),
            ):
        cpu_onnx.run(job)

    store.assert_called_once()
    assert store.call_args.args[0] is job
    assert store.call_args.kwargs["sample_rate"] == 48000
    assert job.status == "done"


def test_external_generation_create_is_idempotent_and_token_is_not_in_repr():
    capability = DeliveryCapability(
        generation_id=uuid4(), capability_token="private-" + "x" * 43,
    )
    manager = JobManager()

    with mock.patch.object(jobs_repo, "get_by_generation", return_value=None), \
            mock.patch.object(jobs_repo, "save") as save, \
            mock.patch("src.services.jobs.threading.Thread") as thread:
        first = manager.create(
            "xin chao", None, "tu_nhien", 0.8, 256,
            user_ref="user-1", delivery=capability,
        )
        second = manager.create(
            "different retry body", None, "tu_nhien", 0.8, 256,
            user_ref="user-1", delivery=capability,
        )

    assert second is first
    assert first.delivery_kind == "openvoice"
    assert first.external_generation_id == str(capability.generation_id)
    assert capability.capability_token not in repr(first)
    save.assert_called_once_with(first)
    thread.assert_called_once()


def test_job_repository_persists_identity_but_not_capability():
    capability = DeliveryCapability(
        generation_id=uuid4(), capability_token="private-" + "x" * 43,
    )
    job = Job(
        id=str(uuid4()), text="xin chao", voice=None, style="tu_nhien",
        temperature=0.8, max_chars=256, delivery=capability,
        delivery_kind="openvoice", external_generation_id=str(capability.generation_id),
    )
    connection = mock.MagicMock()
    context = mock.MagicMock()
    context.__enter__.return_value = connection
    context.__exit__.return_value = False

    with mock.patch.object(jobs_repo, "connect", return_value=context):
        jobs_repo.save(job)

    params = connection.execute.call_args.args[1]
    assert "openvoice" in params
    assert str(capability.generation_id) in params
    assert capability.capability_token not in params
    assert "external_generation_id" in connection.execute.call_args.args[0]
    assert "idx_jobs_external_generation" in db._SCHEMA
