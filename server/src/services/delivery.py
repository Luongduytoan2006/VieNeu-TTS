from __future__ import annotations

import base64
import hashlib
import ipaddress
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional
from urllib.parse import urlsplit

import requests
from pydantic import ValidationError

from ..config import settings
from ..schemas import DeliveryComplete, DeliveryGrant


@dataclass(frozen=True)
class DeliveryResult:
    size: int
    etag: Optional[str]
    confirmed: bool


class DeliveryError(RuntimeError):
    pass


_http = requests.Session()
_http.trust_env = False


def _session() -> requests.Session:
    return _http


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _origin(raw_url: str) -> Optional[str]:
    try:
        parsed = urlsplit(raw_url)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if not host or parsed.username or parsed.password or parsed.fragment:
        return None
    host = host.lower()
    if ":" in host:
        host = f"[{host}]"
    return f"{parsed.scheme.lower()}://{host}{f':{port}' if port else ''}"


def _private_host(hostname: str) -> bool:
    hostname = hostname.lower().rstrip(".")
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return True
    if hostname in {"metadata.google.internal", "host.docker.internal"}:
        return True
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return not address.is_global


def _allowed_upload_url(raw_url: str) -> bool:
    try:
        parsed = urlsplit(raw_url)
        port = parsed.port
    except ValueError:
        return False
    origin = _origin(raw_url)
    base_origin = _origin(settings.OPENVOICE_BASE_URL)
    r2_origin = _origin(settings.OPENVOICE_R2_UPLOAD_ORIGIN)
    if not origin or not parsed.hostname:
        return False

    if parsed.scheme == "http":
        return bool(
            settings.OPENVOICE_ALLOW_HTTP
            and origin == base_origin
            and origin in {"http://localhost:3011", "http://127.0.0.1:3011",
                           "http://host.docker.internal:3011"}
        )

    if parsed.scheme != "https" or port not in (None, 443):
        return False
    if _private_host(parsed.hostname):
        return False
    return origin in {base_origin, r2_origin}


def _retryable(status_code: int) -> bool:
    return status_code in {408, 425, 429} or status_code >= 500


def _request_grant(job, size: int, content_md5: str) -> DeliveryGrant:
    endpoint = (
        f"{settings.OPENVOICE_BASE_URL}/api/voice/internal/deliveries/"
        f"{job.delivery.generation_id}/grant"
    )
    attempts = max(1, settings.OPENVOICE_CALLBACK_RETRIES)
    for attempt in range(attempts):
        try:
            response = _session().post(
                endpoint,
                headers=_headers(job.delivery.capability_token),
                json={
                    "upstream_job_id": job.id,
                    "audio_size_bytes": size,
                    "content_md5": content_md5,
                },
                timeout=settings.OPENVOICE_DELIVERY_TIMEOUT,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            if attempt + 1 >= attempts:
                raise DeliveryError("upload grant transport failed") from exc
            time.sleep(2 ** attempt)
            continue
        if _retryable(response.status_code) and attempt + 1 < attempts:
            time.sleep(2 ** attempt)
            continue
        if response.status_code != 200:
            raise DeliveryError(f"upload grant failed ({response.status_code})")
        try:
            payload = response.json()
            grant = DeliveryGrant.model_validate(payload)
        except (ValueError, ValidationError):
            raise DeliveryError("upload grant response was invalid") from None
        if not _allowed_upload_url(payload["url"]):
            raise DeliveryError("upload grant origin is not allowed")
        now = datetime.now(timezone.utc)
        if grant.expires_at <= now:
            if attempt + 1 < attempts:
                continue
            raise DeliveryError("upload grant expiry is invalid")
        if grant.expires_at > now + timedelta(minutes=15):
            raise DeliveryError("upload grant expiry is invalid")
        if grant.headers["Content-MD5"] != content_md5:
            raise DeliveryError("upload grant checksum does not match")
        return grant

    raise DeliveryError("upload grant transport failed")


CompletionOutcome = Literal["confirmed", "absent", "unknown"]
_OBJECT_VERIFICATION_FAILURE = "Delivered audio failed object verification"


def _completion_confirms_absence(response: requests.Response) -> bool:
    if response.status_code != 409:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    return isinstance(payload, dict) and payload.get("statusMessage") == _OBJECT_VERIFICATION_FAILURE


def _post_completion(job, completion: DeliveryComplete) -> CompletionOutcome:
    endpoint = (
        f"{settings.OPENVOICE_BASE_URL}/api/voice/internal/deliveries/"
        f"{job.delivery.generation_id}/complete"
    )
    attempts = max(1, settings.OPENVOICE_CALLBACK_RETRIES)
    for attempt in range(attempts):
        try:
            response = _session().post(
                endpoint,
                headers=_headers(job.delivery.capability_token),
                json=completion.model_dump(mode="json"),
                timeout=settings.OPENVOICE_DELIVERY_TIMEOUT,
                allow_redirects=False,
            )
        except requests.RequestException:
            response = None
        if response is not None and 200 <= response.status_code < 300:
            return "confirmed"
        if response is not None and _completion_confirms_absence(response):
            return "absent"
        if response is not None and not _retryable(response.status_code):
            return "unknown"
        if attempt + 1 < attempts:
            time.sleep(2 ** attempt)
    return "unknown"


def _notify_error(job, detail: str, elapsed_sec: Optional[float]) -> None:
    try:
        completion = DeliveryComplete(
            upstream_job_id=job.id,
            status="error",
            audio_size_bytes=None,
            object_etag=None,
            duration_sec=None,
            elapsed_sec=elapsed_sec,
            sample_rate=None,
            error=detail[:2000] or "delivery failed",
        )
        _post_completion(job, completion)
    except (TypeError, ValueError):
        pass


def _ready_completion(job, size: int, etag: Optional[str],
                      duration_sec: Optional[float], elapsed_sec: Optional[float],
                      sample_rate: Optional[int]) -> DeliveryComplete:
    return DeliveryComplete(
        upstream_job_id=job.id,
        status="ready",
        audio_size_bytes=size,
        object_etag=etag,
        duration_sec=duration_sec,
        elapsed_sec=elapsed_sec,
        sample_rate=sample_rate,
        error=None,
    )


def deliver(job, wav: bytes, *, duration_sec: Optional[float],
            elapsed_sec: Optional[float], sample_rate: Optional[int]) -> DeliveryResult:
    if not job.delivery:
        raise DeliveryError("missing delivery capability")
    if len(wav) <= 0 or len(wav) > settings.MAX_DELIVERY_BYTES:
        error = DeliveryError("generated WAV has invalid size")
        _notify_error(job, str(error), elapsed_sec)
        raise error

    content_md5 = base64.b64encode(
        hashlib.md5(wav, usedforsecurity=False).digest(),
    ).decode("ascii")
    try:
        grant = _request_grant(job, len(wav), content_md5)
        upload = None
        attempts = max(1, settings.OPENVOICE_CALLBACK_RETRIES)
        for attempt in range(attempts):
            try:
                upload_headers = {
                    "Content-Type": grant.headers["Content-Type"],
                    "Content-MD5": grant.headers["Content-MD5"],
                    "If-None-Match": grant.headers["If-None-Match"],
                }
                if _origin(grant.url) == _origin(settings.OPENVOICE_BASE_URL):
                    upload_headers.update(_headers(job.delivery.capability_token))
                upload = _session().put(
                    grant.url,
                    headers=upload_headers,
                    data=wav,
                    timeout=settings.OPENVOICE_DELIVERY_TIMEOUT,
                    allow_redirects=False,
                )
            except requests.RequestException:
                completion = _ready_completion(
                    job, len(wav), None, duration_sec, elapsed_sec, sample_rate,
                )
                outcome = _post_completion(job, completion)
                if outcome == "confirmed":
                    return DeliveryResult(size=len(wav), etag=None, confirmed=True)
                if outcome == "unknown":
                    return DeliveryResult(size=len(wav), etag=None, confirmed=False)
                if attempt + 1 < attempts:
                    grant = _request_grant(job, len(wav), content_md5)
                    continue
                break
            if 200 <= upload.status_code < 300:
                break
            if upload.status_code == 412:
                completion = _ready_completion(
                    job, len(wav), None, duration_sec, elapsed_sec, sample_rate,
                )
                outcome = _post_completion(job, completion)
                return DeliveryResult(
                    size=len(wav), etag=None, confirmed=outcome == "confirmed",
                )
            if upload.status_code == 403 and attempt + 1 < attempts:
                grant = _request_grant(job, len(wav), content_md5)
                continue
            if _retryable(upload.status_code) and attempt + 1 < attempts:
                time.sleep(2 ** attempt)
                continue
            break
    except DeliveryError as error:
        _notify_error(job, str(error), elapsed_sec)
        raise

    if upload is None or not 200 <= upload.status_code < 300:
        status_code = upload.status_code if upload is not None else 502
        error = DeliveryError(f"audio upload failed ({status_code})")
        _notify_error(job, str(error), elapsed_sec)
        raise error

    etag = upload.headers.get("ETag") or upload.headers.get("etag")
    completion = _ready_completion(
        job, len(wav), etag, duration_sec, elapsed_sec, sample_rate,
    )
    confirmed = _post_completion(job, completion) == "confirmed"
    return DeliveryResult(
        size=len(wav),
        etag=etag,
        confirmed=confirmed,
    )
