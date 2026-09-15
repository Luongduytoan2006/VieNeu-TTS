from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

_SERVER = Path(__file__).resolve().parents[1] / "server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))

from src.schemas import (  # noqa: E402
    MAX_DELIVERY_BYTES,
    DeliveryCapability,
    DeliveryComplete,
    DeliveryGrant,
    JobCreated,
    JobStatus,
    TTSCreate,
)

VALID_MD5 = "1B2M2Y8AsgTpgAmY7PhCfg=="
VALID_HEADERS = {
    "Content-Type": "audio/wav",
    "Content-MD5": VALID_MD5,
    "If-None-Match": "*",
}


def _grant(**overrides) -> dict:
    payload = {
        "method": "PUT",
        "url": "https://example.r2.cloudflarestorage.com/bucket/a%2Fb.wav?b=2&a=1%2B2",
        "headers": VALID_HEADERS,
        "expires_at": "2026-09-14T13:00:00Z",
    }
    payload.update(overrides)
    return payload


def _completion(**overrides) -> dict:
    payload = {
        "upstream_job_id": str(uuid4()),
        "status": "ready",
        "audio_size_bytes": 1024,
        "object_etag": '"etag"',
        "duration_sec": 1.25,
        "elapsed_sec": 2.5,
        "sample_rate": 48000,
        "error": None,
    }
    payload.update(overrides)
    return payload


def test_tts_delivery_is_optional_for_legacy_callers():
    request = TTSCreate(text="Xin chao")

    assert request.delivery is None


def test_delivery_capability_accepts_uuid_and_bounded_token():
    generation_id = uuid4()
    request = TTSCreate(text="Xin chao", delivery={
        "generation_id": str(generation_id),
        "capability_token": "x" * 43,
    })

    assert request.delivery == DeliveryCapability(
        generation_id=generation_id,
        capability_token="x" * 43,
    )
    assert request.model_dump(mode="json")["delivery"]["generation_id"] == str(generation_id)


@pytest.mark.parametrize("generation_id", ["not-a-uuid", "0" * 36])
def test_delivery_capability_rejects_invalid_uuid(generation_id: str):
    with pytest.raises(ValidationError):
        DeliveryCapability(generation_id=generation_id, capability_token="x" * 43)


@pytest.mark.parametrize("token", ["x" * 42, "x" * 129])
def test_delivery_capability_rejects_token_outside_bounds(token: str):
    with pytest.raises(ValidationError):
        DeliveryCapability(generation_id=uuid4(), capability_token=token)


def test_delivery_grant_preserves_raw_signed_url_exactly():
    raw_url = "https://EXAMPLE.r2.cloudflarestorage.com/bucket/a%2Fb.wav?b=2&a=1%2B2"

    grant = DeliveryGrant.model_validate(_grant(url=raw_url))

    assert isinstance(grant.url, str)
    assert grant.url == raw_url
    assert grant.model_dump(mode="json")["url"] == raw_url


@pytest.mark.parametrize("url", [
    "not-a-url",
    "javascript:alert(1)",
    "https:///missing-host",
    "https://user:password@example.test/audio.wav",
    "https://example.test/audio.wav#fragment",
])
def test_delivery_grant_rejects_invalid_or_unsafe_url(url: str):
    with pytest.raises(ValidationError):
        DeliveryGrant.model_validate(_grant(url=url))


def test_delivery_grant_accepts_only_exact_upload_headers():
    grant = DeliveryGrant.model_validate(_grant())

    assert grant.headers == VALID_HEADERS


@pytest.mark.parametrize("headers", [
    {**VALID_HEADERS, "Authorization": "secret"},
    {key: value for key, value in VALID_HEADERS.items() if key != "If-None-Match"},
    {**VALID_HEADERS, "Content-Type": "application/octet-stream"},
    {**VALID_HEADERS, "Content-MD5": "not-base64-md5"},
    {**VALID_HEADERS, "If-None-Match": '"etag"'},
])
def test_delivery_grant_rejects_unknown_missing_or_invalid_headers(headers: dict[str, str]):
    with pytest.raises(ValidationError):
        DeliveryGrant.model_validate(_grant(headers=headers))


def test_delivery_grant_bounds_url_and_requires_put():
    with pytest.raises(ValidationError):
        DeliveryGrant.model_validate(_grant(url="https://example.test/" + "x" * 4090))
    with pytest.raises(ValidationError):
        DeliveryGrant.model_validate(_grant(method="POST"))


def test_ready_completion_requires_audio_metadata_and_no_error():
    completion = DeliveryComplete.model_validate(_completion())

    assert isinstance(completion.upstream_job_id, UUID)
    assert completion.status == "ready"

    for field in ("audio_size_bytes", "duration_sec", "sample_rate"):
        with pytest.raises(ValidationError):
            DeliveryComplete.model_validate(_completion(**{field: None}))
    with pytest.raises(ValidationError):
        DeliveryComplete.model_validate(_completion(error="unexpected"))


def test_error_completion_requires_error_and_null_audio_metadata():
    completion = DeliveryComplete.model_validate(_completion(
        status="error",
        audio_size_bytes=None,
        object_etag=None,
        duration_sec=None,
        sample_rate=None,
        error="synthesis failed",
    ))

    assert completion.status == "error"
    assert completion.elapsed_sec == 2.5

    with pytest.raises(ValidationError):
        DeliveryComplete.model_validate(_completion(
            status="error", audio_size_bytes=None, object_etag=None,
            duration_sec=None, sample_rate=None, error=None,
        ))
    with pytest.raises(ValidationError):
        DeliveryComplete.model_validate(_completion(status="error", error="failed"))


def test_completion_fields_are_bounded():
    with pytest.raises(ValidationError):
        DeliveryComplete.model_validate(_completion(audio_size_bytes=MAX_DELIVERY_BYTES + 1))
    with pytest.raises(ValidationError):
        DeliveryComplete.model_validate(_completion(object_etag="x" * 257))
    with pytest.raises(ValidationError):
        DeliveryComplete.model_validate(_completion(
            status="error", audio_size_bytes=None, object_etag=None,
            duration_sec=None, sample_rate=None, error="x" * 2001,
        ))


def test_job_download_urls_are_nullable_in_json_schema():
    for model in (JobCreated, JobStatus):
        property_schema = model.model_json_schema()["properties"]["download_url"]
        types = {entry.get("type") for entry in property_schema["anyOf"]}
        assert types == {"string", "null"}


def test_delivery_grant_keeps_timezone_aware_expiry():
    expiry = datetime(2026, 9, 14, 13, 0, tzinfo=timezone.utc)

    grant = DeliveryGrant.model_validate(_grant(expires_at=expiry))

    assert grant.expires_at == expiry
