"""test_routing — kiểm tra phân luồng CPU/GPU (unit, OFFLINE).

Test thuần logic ``resolve_mode`` + ``count_words`` trong
``server/src/services/createVoice.py``. KHÔNG chạm model / Vast.ai / mạng.

Chạy được cả 2 cách:
  * pytest:  ../.venv/Scripts/python.exe -m pytest tests/test_routing.py -v
  * python:  ../.venv/Scripts/python.exe tests/test_routing.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Bootstrap: thêm server/ vào sys.path để ``import src...`` hoạt động ──────────
_SERVER = Path(__file__).resolve().parents[1] / "server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))

# Đặt TRƯỚC khi import src.config để tránh eager-load model ở các test file khác
# khi pytest gom chung 1 tiến trình.
os.environ.setdefault("MODEL_EAGER_LOAD", "0")
os.environ.setdefault("STORAGE_BACKEND", "local")

from src.config import settings  # noqa: E402
from src.services.createVoice import CreateError, count_words, resolve_mode  # noqa: E402

# 1000 từ (tách theo khoảng trắng) → đạt ngưỡng GPU mặc định (GPU_MIN_WORDS=1000).
LONG_TEXT = " ".join(["từ"] * 1000)
SHORT_TEXT = "xin chào thế giới"          # 3 từ


# ── count_words ────────────────────────────────────────────────────────────────
def test_count_words_basic():
    assert count_words("a b c") == 3
    assert count_words("một") == 1
    assert count_words("") == 0
    # Khoảng trắng thừa / xuống dòng không làm sai số từ.
    assert count_words("  a   b  \n c ") == 3
    assert count_words(LONG_TEXT) == 1000


# ── resolve_mode: auto ─────────────────────────────────────────────────────────
def test_auto_short_to_cpu():
    """Văn bản ngắn + auto → cpu."""
    assert resolve_mode(SHORT_TEXT, "auto") == "cpu"


def test_auto_long_to_gpu():
    """Văn bản ≥ ngưỡng từ + auto → gpu."""
    assert resolve_mode(LONG_TEXT, "auto") == "gpu"


def test_empty_mode_defaults_auto():
    """mode rỗng/None → auto → theo độ dài."""
    assert resolve_mode(SHORT_TEXT, "") == "cpu"
    assert resolve_mode(LONG_TEXT, "") == "gpu"


# ── resolve_mode: gpu (lớp chặn BE thứ 2) ──────────────────────────────────────
def test_gpu_short_blocked_422():
    """Ngắn + gpu → CHẶN bằng CreateError(422)."""
    try:
        resolve_mode(SHORT_TEXT, "gpu")
    except CreateError as e:
        assert e.status == 422
        # Thông điệp nêu ngưỡng số từ.
        assert str(settings.GPU_MIN_WORDS) in e.detail
    else:
        raise AssertionError("resolve_mode ngắn+gpu phải raise CreateError(422)")


def test_gpu_long_ok():
    """Dài + gpu → gpu (đủ điều kiện)."""
    assert resolve_mode(LONG_TEXT, "gpu") == "gpu"


def test_gpu_uppercase_normalized():
    """mode được lower() trước khi so — 'GPU' ngắn vẫn bị chặn 422."""
    try:
        resolve_mode(SHORT_TEXT, "GPU")
    except CreateError as e:
        assert e.status == 422
    else:
        raise AssertionError("'GPU' (hoa) ngắn phải raise 422")


# ── resolve_mode: cpu (luôn cpu) ───────────────────────────────────────────────
def test_cpu_always_cpu():
    """cpu luôn cpu — kể cả văn bản dài."""
    assert resolve_mode(SHORT_TEXT, "cpu") == "cpu"
    assert resolve_mode(LONG_TEXT, "cpu") == "cpu"


# ── resolve_mode: mode lạ ──────────────────────────────────────────────────────
def test_unknown_mode_422():
    """mode không thuộc {cpu,gpu,auto} → CreateError(422)."""
    try:
        resolve_mode(SHORT_TEXT, "xxx")
    except CreateError as e:
        assert e.status == 422
    else:
        raise AssertionError("mode lạ phải raise CreateError(422)")


# ── Ngưỡng biên: ngay dưới ngưỡng + gpu bị chặn ────────────────────────────────
def test_gpu_just_below_threshold_blocked():
    n = settings.GPU_MIN_WORDS - 1
    text = " ".join(["từ"] * n)
    assert count_words(text) == n
    try:
        resolve_mode(text, "gpu")
    except CreateError as e:
        assert e.status == 422
    else:
        raise AssertionError(f"{n} từ (dưới ngưỡng) + gpu phải raise 422")


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
