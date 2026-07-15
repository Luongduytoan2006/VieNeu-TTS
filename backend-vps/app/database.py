"""SQLAlchemy (sync) engine + session factory.

Sync on purpose: job workers run in threads (not async), so a sync ORM is the
simplest correct fit. Each unit of work opens a short-lived Session via the
``session_scope`` context manager.
"""
from __future__ import annotations

import contextlib
import time
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


def _make_engine():
    """Build the engine. Supports Postgres (production) and SQLite (zero-infra:
    lets the VPS run with just ``uv``, no Postgres container). SQLite needs
    ``check_same_thread=False`` because job workers run in threads."""
    url = settings.DATABASE_URL
    if url.startswith("sqlite"):
        return create_engine(
            url, future=True,
            connect_args={"check_same_thread": False},
        )
    return create_engine(url, pool_pre_ping=True, future=True)


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


def init_db(retries: int = 10, delay: float = 1.5) -> None:
    """Create tables, waiting for Postgres to accept connections first."""
    from . import models  # noqa: F401  (register mappers)

    last_err: Exception | None = None
    for _ in range(retries):
        try:
            Base.metadata.create_all(engine)
            return
        except Exception as e:  # DB not up yet
            last_err = e
            time.sleep(delay)
    raise RuntimeError(f"Could not initialize database: {last_err}")


@contextlib.contextmanager
def session_scope() -> Iterator[Session]:
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
