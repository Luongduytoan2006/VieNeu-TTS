"""Pluggable object storage for generated audio.

The model-server writes finished audio here and hands back a URL the BE/client can
fetch. Two backends:

* ``local``  — files on disk, served by the model-server's ``/files/{key}`` route
               (fine for single-machine dev / testing).
* ``r2``     — Cloudflare R2 (S3-compatible). The GPU box uploads the WAV and returns
               a **presigned GET URL**, so the BE (or client) can download it over
               plain HTTP without any credentials. This is the split-deploy default.

Select with ``STORAGE_BACKEND`` (``local`` | ``r2``). R2 needs
``R2_URL`` / ``R2_ACCESS_KEY`` / ``R2_SECRET_KEY`` / ``R2_BUCKET``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Protocol

logger = logging.getLogger("Vieneu.ModelServer.storage")


class Storage(Protocol):
    def put(self, key: str, data: bytes, content_type: str = "audio/wav") -> str:
        """Store bytes under ``key``; return a URL the BE/client can fetch."""
        ...

    def url_for(self, key: str) -> str:
        ...

    def open(self, key: str) -> bytes:
        ...

    def exists(self, key: str) -> bool:
        ...

    def delete(self, key: str) -> None:
        ...


class LocalStorage:
    """Files under a directory, served back via the model-server's /files/{key}.

    ``public_base`` is the externally reachable base URL of this server (what the
    BE/client should hit), e.g. http://localhost:9000. The returned URL is
    ``{public_base}/files/{key}``.
    """

    def __init__(self, root: str | os.PathLike, public_base: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.public_base = public_base.rstrip("/")

    def _path(self, key: str) -> Path:
        # keep it flat + safe: no traversal
        safe = key.replace("..", "_").lstrip("/")
        p = self.root / safe
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def put(self, key: str, data: bytes, content_type: str = "audio/wav") -> str:
        self._path(key).write_bytes(data)
        return self.url_for(key)

    def url_for(self, key: str) -> str:
        return f"{self.public_base}/files/{key}"

    def open(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()


class R2Storage:
    """Cloudflare R2 (S3-compatible) via boto3.

    ``put`` uploads the object and returns a presigned GET URL valid for
    ``presign_ttl`` seconds — the BE downloads it with a plain HTTP GET, no creds.
    If ``public_base`` is set (an R2 public bucket / custom domain) that permanent
    public URL is used instead of a presigned one.
    """

    def __init__(self, *, endpoint: str, access_key: str, secret_key: str,
                 bucket: str, public_base: Optional[str] = None,
                 presign_ttl: int = 86400) -> None:
        import boto3
        from botocore.config import Config

        self.bucket = bucket
        self.public_base = public_base.rstrip("/") if public_base else None
        self.presign_ttl = presign_ttl
        self._s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
            config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        from botocore.exceptions import ClientError
        try:
            self._s3.head_bucket(Bucket=self.bucket)
        except ClientError:
            try:
                self._s3.create_bucket(Bucket=self.bucket)
                logger.info("🪣 R2 bucket '%s' created", self.bucket)
            except ClientError as e:
                logger.warning("R2 bucket check/create failed for '%s': %s", self.bucket, e)

    def put(self, key: str, data: bytes, content_type: str = "audio/wav") -> str:
        key = key.lstrip("/")
        self._s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        return self.url_for(key)

    def url_for(self, key: str) -> str:
        key = key.lstrip("/")
        if self.public_base:
            return f"{self.public_base}/{key}"
        return self._s3.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=self.presign_ttl,
        )

    def open(self, key: str) -> bytes:
        return self._s3.get_object(Bucket=self.bucket, Key=key.lstrip("/"))["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self._s3.head_object(Bucket=self.bucket, Key=key.lstrip("/"))
            return True
        except ClientError:
            return False

    def delete(self, key: str) -> None:
        self._s3.delete_object(Bucket=self.bucket, Key=key.lstrip("/"))


def make_storage() -> Storage:
    """Build the storage backend from env. Defaults to local disk."""
    backend = os.getenv("STORAGE_BACKEND", "local").lower()

    if backend in ("r2", "s3"):
        endpoint = os.getenv("R2_URL") or os.getenv("R2_ENDPOINT")
        access = os.getenv("R2_ACCESS_KEY")
        secret = os.getenv("R2_SECRET_KEY")
        bucket = os.getenv("R2_BUCKET", "vieneu-tts")
        public_base = os.getenv("R2_PUBLIC_BASE") or None
        ttl = int(os.getenv("R2_PRESIGN_TTL", "86400"))
        missing = [n for n, v in (("R2_URL", endpoint), ("R2_ACCESS_KEY", access),
                                  ("R2_SECRET_KEY", secret)) if not v]
        if missing:
            raise RuntimeError(f"STORAGE_BACKEND={backend} but missing env: {', '.join(missing)}")
        logger.info("🗄️  storage=R2 bucket=%s (%s)", bucket, "public" if public_base else "presigned")
        return R2Storage(endpoint=endpoint, access_key=access, secret_key=secret,
                         bucket=bucket, public_base=public_base, presign_ttl=ttl)

    if backend == "local":
        root = os.getenv("STORAGE_LOCAL_DIR")
        if not root:
            # repo .ask/storage by default (three levels up from this file)
            root = str(Path(__file__).resolve().parents[2] / ".ask" / "storage")
        public = os.getenv("MODEL_SERVER_PUBLIC_URL", "http://localhost:9000")
        logger.info("🗄️  storage=local dir=%s public=%s", root, public)
        return LocalStorage(root, public)

    raise NotImplementedError(f"Storage backend '{backend}' chưa hỗ trợ (dùng: local | r2).")
