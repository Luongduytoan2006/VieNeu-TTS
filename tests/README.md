# VieNeu-TTS Testing Directory

This directory contains test suites and utilities for verifying the VieNeu-TTS package.

Two families of tests live here:

1. **Backend API tests** (`test_routing.py`, `test_api_cpu.py`, `test_vastai_flow.py`) —
   fast, **fully offline/mocked**, target the `server/` FastAPI backend. No model,
   no Vast.ai, no R2, **no cost**. These are the ones described in the next section.
2. **Model-SDK tests** (`test_engine_*.py`, `test_factory.py`, `test_utils.py`) — exercise
   the `vieneu` engine itself; may download weights / need a GPU. Documented further down.

---

## Backend tests (offline, no cost) — run these

These verify the split backend in `server/` (CPU/GPU routing, HTTP contract, Vast.ai
client quirks) **without** touching the model, Vast.ai, or R2. Every test is offline
and mocked — safe to run anywhere.

> **Note:** the repo ships a single virtualenv at the **repo root** (`.venv/`), not under
> `server/`. `pytest` is **not** installed there, so the tests are written to run both under
> pytest *and* as plain scripts (`python tests/test_x.py`). Each file sets `sys.path` to
> include `server/` and sets `MODEL_EAGER_LOAD=0` / `STORAGE_BACKEND=local` before importing,
> so no extra env setup is needed.

### Run with plain python (works today — no pytest needed)

From the repo root:

```bash
.venv/Scripts/python.exe tests/test_routing.py
.venv/Scripts/python.exe tests/test_api_cpu.py
.venv/Scripts/python.exe tests/test_vastai_flow.py
```

Each prints `PASS <name>` lines and ends with `ALL PASSED` (exit code 0) or `N FAILED`.

### Run with pytest (if/when installed)

```bash
# install once:  .venv/Scripts/python.exe -m pip install pytest
cd tests && ../.venv/Scripts/python.exe -m pytest -v
```

The same `def test_*` functions are pytest-compatible.

### What each backend test covers

- **`test_routing.py`** — unit tests for `resolve_mode` + `count_words`
  (`server/src/services/createVoice.py`). Short + `auto` → `cpu`; ≥ `GPU_MIN_WORDS`
  words + `auto` → `gpu`; short + `gpu` → `CreateError(422)`; long + `gpu` → `gpu`;
  `cpu` always `cpu`; unknown mode → `422`; word counting + threshold boundary.
- **`test_api_cpu.py`** — `fastapi.testclient.TestClient` against `server/main.py:app`
  with `MODEL_EAGER_LOAD=0` (no model load). `GET /api/v1/health` → 200 with
  `gpu_min_words`; `GET /api/v1/modes` → 200 lists `cpu`+`gpu`; `GET /api/v1/styles` → 200;
  `POST /api/v1/tts` when model not loaded → 503; short text + `mode=gpu` (model faked
  loaded) → 422 (backend GPU guard).
- **`test_vastai_flow.py`** — `VastAIClient` (`server/src/services/gpu_vastai.py`) with
  **`requests.request` mocked** — never hits the real API. Missing `VAST_AI_API_KEY`
  → `VastAIError`; `search_offer` parses `offers[]` (cheapest first, PUT to `/api/v0/`);
  `create` reads `new_contract` (not `id`); `destroy` uses `DELETE` to `API_V0`
  (`/api/v0/instances/{id}/`) and verifies.

### Live GPU test (real Vast.ai, **costs money — not run here**)

The end-to-end GPU smoke test that actually rents a machine lives in the POC script
`.ask/scripts/vast_poc.py` (create → synth → download → destroy, billed by the second).
It is **not** part of this suite and is only mentioned for reference — do not run it as
part of `tests/`.

---

## Model-SDK tests

These target the `vieneu` engine and may download weights or require a GPU. Ensure you
are in the project root:

```bash
uv run pytest
```

This will automatically discover and run all test suites in the `tests/` directory.

---

### Individual Test Suites
- **[test_engine_standard.py](test_engine_standard.py)**: Tests for the standard VieNeuTTS engine (Torch/GGUF).
- **[test_engine_remote.py](test_engine_remote.py)**: Tests for the Remote API engine.
- **[test_engine_fast.py](test_engine_fast.py)**: Tests for the Fast (LMDeploy) engine.
- **[test_factory.py](test_factory.py)**: Tests for the Vieneu factory class.
- **[test_utils.py](test_utils.py)**: Tests for audio and text processing utilities.

---

### Other Utilities
- **[benchmark.py](benchmark.py)**: RTF and latency benchmarking.
