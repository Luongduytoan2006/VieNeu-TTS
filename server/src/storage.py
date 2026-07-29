"""Object storage cho audio kết quả — chọn backend qua STORAGE_BACKEND.

* ``local`` — file trên đĩa, phục vụ qua route ``/files/{key}`` của controller
              (CPU mode / single-machine dev).
* ``r2``    — Cloudflare R2 (S3-compatible). Upload WAV rồi trả **presigned GET URL**
              để client tải qua HTTP thường, không cần creds. Đây là đường GPU/prod
              (server GPU ephemeral upload lên R2 rồi destroy).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Protocol

from .config import settings

logger = logging.getLogger("Vieneu.Storage")


class Storage(Protocol):
    def put(self, key: str, data: bytes, content_type: str = "audio/wav") -> str: ...
    def url_for(self, key: str) -> str: ...
    def open(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...


class LocalStorage:
    """File dưới 1 thư mục, trả về qua ``{public_base}/files/{key}``."""

    def __init__(self, root: str, public_base: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.public_base = public_base.rstrip("/")

    def _path(self, key: str) -> Path:
        safe = key.replace("..", "_").lstrip("/")
        p = self.root / safe
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def put(self, key: str, data: bytes, content_type: str = "audio/wav") -> str:
        self._path(key).write_bytes(data)
        return self.url_for(key)

    def url_for(self, key: str) -> str:
        return f"{self.public_base}/files/{key.lstrip('/')}"

    def open(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()


class R2Storage:
    """Cloudflare R2 (S3-compatible) qua boto3.

    ``put`` upload object và trả presigned GET URL sống ``presign_ttl`` giây. Nếu có
    ``public_base`` (bucket công khai / custom domain) thì dùng URL vĩnh viễn đó.
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
            "s3", endpoint_url=endpoint,
            aws_access_key_id=access_key, aws_secret_access_key=secret_key,
            region_name="auto",
            config=Config(signature_version="s3v4",
                          retries={"max_attempts": 3, "mode": "standard"}),
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
            ExpiresIn=self.presign_ttl)

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
    """Dựng backend storage từ settings. Mặc định local disk."""
    backend = settings.STORAGE_BACKEND

    if backend in ("r2", "s3"):
        missing = [n for n, v in (("R2_URL", settings.R2_URL),
                                  ("R2_ACCESS_KEY", settings.R2_ACCESS_KEY),
                                  ("R2_SECRET_KEY", settings.R2_SECRET_KEY)) if not v]
        if missing:
            raise RuntimeError(f"STORAGE_BACKEND={backend} nhưng thiếu env: {', '.join(missing)}")
        logger.info("🗄️  storage=R2 bucket=%s (%s)", settings.R2_BUCKET,
                    "public" if settings.R2_PUBLIC_BASE else "presigned")
        return R2Storage(endpoint=settings.R2_URL, access_key=settings.R2_ACCESS_KEY,
                         secret_key=settings.R2_SECRET_KEY, bucket=settings.R2_BUCKET,
                         public_base=settings.R2_PUBLIC_BASE or None,
                         presign_ttl=settings.R2_PRESIGN_TTL)

    if backend == "local":
        logger.info("🗄️  storage=local dir=%s public=%s",
                    settings.STORAGE_LOCAL_DIR, settings.PUBLIC_BASE_URL)
        return LocalStorage(settings.STORAGE_LOCAL_DIR, settings.PUBLIC_BASE_URL)

    raise NotImplementedError(f"Storage backend '{backend}' chưa hỗ trợ (dùng: local | r2).")


# Khởi tạo lười: chỉ dựng khi lần đầu dùng (tránh yêu cầu boto3/R2 khi chỉ chạy CPU local).
_storage: Optional[Storage] = None
_storage_lock = __import__("threading").Lock()


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        with _storage_lock:
            if _storage is None:
                _storage = make_storage()
    return _storage
