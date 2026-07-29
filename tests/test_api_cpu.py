"""test_api_cpu — kiểm tra API HTTP bằng TestClient, KHÔNG tải model thật (OFFLINE).

Đặt ``MODEL_EAGER_LOAD=0`` + ``STORAGE_BACKEND=local`` TRƯỚC khi import ``main`` để
lifespan KHÔNG nạp model (không cần GPU/mạng/HF). Các endpoint tra cứu (health,
modes, styles) trả 200 không cần model. POST /tts:
  * model chưa load           → 503 (chặn ở createVoice.create).
  * model "load" + ngắn + gpu → 422 (lớp chặn GPU thứ 2 ở BE).

Chạy được cả 2 cách:
  * pytest:  ../.venv/Scripts/python.exe -m pytest tests/test_api_cpu.py -v
  * python:  ../.venv/Scripts/python.exe tests/test_api_cpu.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Bootstrap sys.path + env (PHẢI đặt env TRƯỚC khi import main) ────────────────
_SERVER = Path(__file__).resolve().parents[1] / "server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))

os.environ["MODEL_EAGER_LOAD"] = "0"      # KHÔNG eager-load model ở startup
os.environ["STORAGE_BACKEND"] = "local"   # không đụng R2
# DB: dùng DATABASE_URL từ môi trường (Postgres). Test ghi/xóa 1 giọng tạm
# user_ref='_test' nên không đụng dữ liệu thật.

import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from src.engine import engine  # noqa: E402
from src.repositories import voices_repo as repo  # noqa: E402


def _client() -> TestClient:
    return TestClient(main.app)


# ── health ─────────────────────────────────────────────────────────────────────
def test_health_ok_has_gpu_min_words():
    with _client() as c:
        r = c.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert "gpu_min_words" in body
    assert isinstance(body["gpu_min_words"], int)
    # Model không eager-load → status 'loading' (hoặc 'error'), model_loaded False.
    assert body["model_loaded"] is False


# ── modes ──────────────────────────────────────────────────────────────────────
def test_modes_lists_cpu_and_gpu():
    with _client() as c:
        r = c.get("/api/v1/modes")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["modes"]}
    assert {"cpu", "gpu"} <= ids


# ── styles ─────────────────────────────────────────────────────────────────────
def test_styles_ok():
    with _client() as c:
        r = c.get("/api/v1/styles")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    assert "default_style" in body


# ── POST /tts: model chưa load → 503 ───────────────────────────────────────────
def test_tts_returns_503_when_model_not_loaded():
    # Đảm bảo engine ở trạng thái CHƯA load.
    engine._tts = None
    with _client() as c:
        r = c.post("/api/v1/tts", json={"text": "xin chào", "mode": "cpu"})
    assert r.status_code == 503


# ── POST /tts: model "load" + text ngắn + gpu → 422 (BE chặn GPU) ───────────────
def test_tts_short_gpu_blocked_422():
    # Giả lập model đã load để vượt qua check 503. Cần 1 giọng custom trong DB
    # (user _test) để qua bước resolve-voice và thật sự chạm lớp chặn GPU (422).
    import numpy as np
    saved = engine._tts
    engine._tts = object()                       # engine.loaded -> True
    repo.upsert("_test", "_t", speaker_emb=np.zeros(192), codes=None)
    try:
        with _client() as c:
            r = c.post("/api/v1/tts",
                       headers={"X-User-Id": "_test"},
                       json={"text": "xin chào", "mode": "gpu", "voice": "_t"})
        assert r.status_code == 422
        assert "GPU" in r.json()["detail"]       # đúng lỗi gate GPU, không phải voice
    finally:
        engine._tts = saved
        repo.delete("_test", "_t")


# ── Runner khi chạy bằng ``python`` thuần (không có pytest) ─────────────────────
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
