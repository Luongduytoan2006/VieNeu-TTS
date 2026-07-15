"""Pluggable object storage for generated audio.

The model-server writes finished audio here and hands back a URL. Today the only
backend is local disk (a folder, served by the model-server's /files route);
tomorrow this is where an R2/S3/MinIO backend slots in without touching callers.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol


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


def make_storage() -> Storage:
    """Build the storage backend from env. Defaults to local disk in .ask/storage."""
    backend = os.getenv("STORAGE_BACKEND", "local").lower()
    if backend == "local":
        root = os.getenv("STORAGE_LOCAL_DIR")
        if not root:
            # repo .ask/storage by default (three levels up from this file)
            root = str(Path(__file__).resolve().parents[2] / ".ask" / "storage")
        public = os.getenv("MODEL_SERVER_PUBLIC_URL", "http://localhost:9000")
        return LocalStorage(root, public)
    # Placeholder for the future R2/S3 backend.
    raise NotImplementedError(f"Storage backend '{backend}' chưa hỗ trợ (dự kiến: r2/s3).")
