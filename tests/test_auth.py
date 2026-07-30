"""test_auth — kiểm tra cổng API key (Bearer) — OFFLINE, KHÔNG model/DB.

Auth = 1 mã dùng chung ``ACCESS_SECRET_KEY``. Middleware ``_RequireApiKey`` chặn
mọi ``/api/v1/*`` TRỪ ``/health`` (và OPTIONS preflight, các path ngoài /api/v1).

Để test ĐỘC LẬP THỨ TỰ import (pytest gom chung 1 tiến trình với các test khác có
thể đã import ``main`` khi auth TẮT), ta KHÔNG dựa vào middleware gắn lúc dựng app.
Thay vào đó bọc thẳng ``main.app`` bằng ``_RequireApiKey`` và patch
``settings.API_KEY`` — middleware đọc key ở runtime nên patch là ăn ngay. Dùng
``TestClient`` KHÔNG ``with`` để khỏi chạy lifespan (không cần Postgres/model): 401
xảy ra ở middleware TRƯỚC khi chạm route.

Chạy được cả 2 cách:
  * pytest:  ../.venv/Scripts/python.exe -m pytest tests/test_auth.py -v
  * python:  ../.venv/Scripts/python.exe tests/test_auth.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

# ── Bootstrap sys.path + env (đặt TRƯỚC khi import main) ─────────────────────────
_SERVER = Path(__file__).resolve().parents[1] / "server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))

os.environ.setdefault("MODEL_EAGER_LOAD", "0")
os.environ.setdefault("STORAGE_BACKEND", "local")

import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from src.config import settings  # noqa: E402

TEST_KEY = "test-secret-key-abc123XYZ"


def _client() -> TestClient:
    """App có cổng key ÉP BẬT (bọc _RequireApiKey quanh main.app). Không lifespan."""
    return TestClient(main._RequireApiKey(main.app))


# ── /health MIỄN key (health-check/monitor gọi được không cần key) ──────────────
def test_health_exempt_no_key():
    # /health chạm DB → mock service để test này độc lập Postgres (chỉ cần chứng minh
    # middleware KHÔNG chặn /health khi thiếu key). Patch tại nơi controller gọi tới.
    from src.services import catalog as catalog_svc
    from src.schemas import HealthResponse
    fake = HealthResponse(status="ok", model_loaded=True, backend="onnx",
                          device="cpu", backbone_repo="x", num_voices=0,
                          gpu_min_words=settings.GPU_MIN_WORDS)
    with mock.patch.object(settings, "API_KEY", TEST_KEY), \
            mock.patch.object(catalog_svc, "health", return_value=fake):
        r = _client().get("/api/v1/health")
    assert r.status_code != 401          # miễn auth (không bị 401 dù thiếu key)
    assert r.status_code == 200


# ── endpoint thường: THIẾU key → 401 ───────────────────────────────────────────
def test_protected_missing_key_401():
    with mock.patch.object(settings, "API_KEY", TEST_KEY):
        r = _client().get("/api/v1/modes")
    assert r.status_code == 401
    assert "key" in r.json()["detail"].lower()


# ── SAI key → 401 ───────────────────────────────────────────────────────────────
def test_protected_wrong_key_401():
    with mock.patch.object(settings, "API_KEY", TEST_KEY):
        r = _client().get("/api/v1/modes",
                          headers={"Authorization": "Bearer sai-be-roi"})
    assert r.status_code == 401


# ── ĐÚNG key (Bearer) → qua cổng (không 401; modes tĩnh nên 200) ────────────────
def test_protected_valid_key_passes():
    with mock.patch.object(settings, "API_KEY", TEST_KEY):
        r = _client().get("/api/v1/modes",
                          headers={"Authorization": f"Bearer {TEST_KEY}"})
    assert r.status_code != 401
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["modes"]}
    assert {"cpu", "gpu"} <= ids


# ── Thiếu tiền tố 'Bearer ' → 401 (chỉ dán key trần không đủ) ────────────────────
def test_raw_key_without_bearer_401():
    with mock.patch.object(settings, "API_KEY", TEST_KEY):
        r = _client().get("/api/v1/modes", headers={"Authorization": TEST_KEY})
    assert r.status_code == 401


# ── OPTIONS (CORS preflight) MIỄN key ───────────────────────────────────────────
def test_options_preflight_exempt():
    with mock.patch.object(settings, "API_KEY", TEST_KEY):
        r = _client().options("/api/v1/modes")
    assert r.status_code != 401


# ── Path NGOÀI /api/v1 (vd '/docs') MIỄN key ────────────────────────────────────
def test_non_api_path_exempt():
    with mock.patch.object(settings, "API_KEY", TEST_KEY):
        r = _client().get("/docs")
    assert r.status_code != 401


# ── Key RỖNG = auth TẮT: endpoint thường qua không cần key ──────────────────────
def test_empty_key_disables_auth():
    with mock.patch.object(settings, "API_KEY", ""):
        r = _client().get("/api/v1/modes")
    assert r.status_code != 401
    assert r.status_code == 200


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
