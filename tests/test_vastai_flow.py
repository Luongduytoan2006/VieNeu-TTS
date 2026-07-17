"""test_vastai_flow — kiểm tra VastAIClient KHÔNG gọi API thật, KHÔNG tốn tiền.

Tất cả HTTP bị MOCK (patch ``gpu_vastai.requests.request``). Xác minh các quirk
Vast.ai đã trả giá bằng tiền thật ở POC:
  * thiếu VAST_AI_API_KEY  → VastAIError ngay khi khởi tạo.
  * search_offer           → parse ``offers[]`` (rẻ nhất = phần tử đầu), dùng PUT.
  * create                 → đọc ``new_contract`` (KHÔNG phải ``id``).
  * destroy                → DELETE tới API_V0 (/api/v0/), verify lại bằng GET.

Chạy được cả 2 cách:
  * pytest:  ../.venv/Scripts/python.exe -m pytest tests/test_vastai_flow.py -v
  * python:  ../.venv/Scripts/python.exe tests/test_vastai_flow.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

# ── Bootstrap sys.path + env ────────────────────────────────────────────────────
_SERVER = Path(__file__).resolve().parents[1] / "server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))

os.environ.setdefault("MODEL_EAGER_LOAD", "0")
os.environ.setdefault("STORAGE_BACKEND", "local")

from src.config import settings  # noqa: E402
from src.services import gpu_vastai  # noqa: E402
from src.services.gpu_vastai import API_V0, VastAIClient, VastAIError  # noqa: E402


# ── Fake response cho requests.request ─────────────────────────────────────────
class _FakeResp:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text
        self.content = b"x" if json_data is not None else b""

    def json(self):
        return self._json


def _client_with_key() -> VastAIClient:
    """Tạo client với key giả (patch settings để không phụ thuộc .env thật)."""
    with mock.patch.object(settings, "VAST_AI_API_KEY", "fake-key-123"):
        return VastAIClient()


# ── (a) thiếu key → VastAIError ────────────────────────────────────────────────
def test_missing_key_raises():
    # settings.VAST_AI_API_KEY được nạp lúc import; patch tạm về "" để mô phỏng thiếu.
    with mock.patch.object(settings, "VAST_AI_API_KEY", ""):
        try:
            VastAIClient()
        except VastAIError:
            pass
        else:
            raise AssertionError("Thiếu VAST_AI_API_KEY phải raise VastAIError")


def test_init_with_key_ok():
    c = _client_with_key()
    assert c.key == "fake-key-123"
    assert c.instance_id is None      # trạng thái vòng đời chưa điền


# ── (b) search_offer parse offers[] (rẻ nhất = phần tử đầu) ─────────────────────
def test_search_offer_parses_offers():
    c = _client_with_key()
    offers = {"offers": [
        {"id": 111, "dph_total": 0.05},
        {"id": 222, "dph_total": 0.09},
    ]}
    with mock.patch.object(gpu_vastai.requests, "request",
                           return_value=_FakeResp(200, offers)) as req:
        offer = c.search_offer()
    assert offer is not None
    assert offer["id"] == 111                     # phần tử đầu = rẻ nhất
    # Quirk: search dùng PUT tới /api/v0/search/asks/.
    method, url = req.call_args.args[0], req.call_args.args[1]
    assert method == "PUT"
    assert url.startswith(API_V0)
    assert "/search/asks/" in url


def test_search_offer_empty_returns_none():
    c = _client_with_key()
    with mock.patch.object(gpu_vastai.requests, "request",
                           return_value=_FakeResp(200, {"offers": []})):
        assert c.search_offer() is None


def test_search_offer_error_raises():
    c = _client_with_key()
    with mock.patch.object(gpu_vastai.requests, "request",
                           return_value=_FakeResp(500, None, text="boom")):
        try:
            c.search_offer()
        except VastAIError:
            pass
        else:
            raise AssertionError("search_offer status!=200 phải raise VastAIError")


# ── (c) create đọc new_contract (KHÔNG phải id) ────────────────────────────────
def test_create_reads_new_contract():
    c = _client_with_key()
    # Cố tình để 'id' khác new_contract — nếu code đọc nhầm 'id' sẽ sai.
    resp = _FakeResp(200, {"new_contract": 987654, "id": 111})
    with mock.patch.object(gpu_vastai.requests, "request", return_value=resp) as req:
        iid = c.create(offer_id=111)
    assert iid == 987654
    assert c.instance_id == 987654
    method, url = req.call_args.args[0], req.call_args.args[1]
    assert method == "PUT"
    assert url == f"{API_V0}/asks/111/"


def test_create_no_contract_raises():
    c = _client_with_key()
    with mock.patch.object(gpu_vastai.requests, "request",
                           return_value=_FakeResp(200, {"id": 111})):  # thiếu new_contract
        try:
            c.create(offer_id=111)
        except VastAIError:
            pass
        else:
            raise AssertionError("create thiếu new_contract phải raise VastAIError")


# ── (d) destroy dùng DELETE tới API_V0 + verify ────────────────────────────────
def test_destroy_uses_delete_v0_and_verifies():
    c = _client_with_key()
    c.instance_id = 987654
    calls = []

    def _fake_request(method, url, **kw):
        calls.append((method, url))
        if method == "DELETE":
            return _FakeResp(200, {"success": True})
        # GET get_instance → máy đã chết (không còn instance).
        return _FakeResp(404, None)

    with mock.patch.object(gpu_vastai.requests, "request", side_effect=_fake_request):
        dead = c.destroy()

    assert dead is True
    # Phải có ít nhất 1 DELETE tới /api/v0/instances/{id}/.
    deletes = [(m, u) for (m, u) in calls if m == "DELETE"]
    assert deletes, "destroy phải gọi DELETE"
    for _m, u in deletes:
        assert u == f"{API_V0}/instances/987654/"
        assert "/api/v0/" in u          # quirk: DESTROY luôn v0 (v1 → 404 nhưng máy vẫn sống)


def test_destroy_no_instance_is_noop_true():
    c = _client_with_key()
    c.instance_id = None
    # Không có instance → coi như đã chết, KHÔNG gọi HTTP.
    with mock.patch.object(gpu_vastai.requests, "request",
                           side_effect=AssertionError("không được gọi HTTP")):
        assert c.destroy() is True


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
