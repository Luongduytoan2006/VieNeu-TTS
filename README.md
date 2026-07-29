# 🦜 VieNeu-TTS — Self-Hosted Server

Self-hosted **Vietnamese-native bilingual (VN/EN) Text-to-Speech**. Built on **VieNeu-TTS v3 Turbo
(48 kHz)** with 10,000+ hours of training data, **instant voice cloning** (3–5 s reference), and
experimental **emotion / non-verbal cues** (`[cười]` laugh, `[thở dài]` sigh). Apache-2.0.

[![Hugging Face v3 Turbo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v3%20Turbo-red)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

> This repository is a **server/deployment build** of VieNeu-TTS: one FastAPI app that runs the model
> in-process on **CPU (torch-free ONNX)** and offloads long jobs to an **on-demand GPU** (Vast.ai).
> For the original SDK / research project, see [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS).

---

## ✨ Highlights

- **v3 Turbo, 48 kHz** — high-fidelity, natural Vietnamese speech.
- **Torch-free on CPU** — the default path runs entirely on ONNX Runtime; PyTorch is never imported.
- **Instant voice cloning** — clone any voice from 3–5 s of audio; the profile is ~4.5 KB and stored in the DB.
- **Built-in default voices** — 10 presets, callable by name, no reference clip needed.
- **Bilingual (En–Vi) code-switching**, fully offline on CPU.
- **Multi-user** — custom voices and jobs are isolated per user (`X-User-Id` header).

---

## 🏗️ Architecture

One server (`server/main.py`) runs the whole app on **a single port** (`7862`) — the deploy target
(CT3600) only exposes one:

| Path on `:7862` | |
|---|---|
| `/` | **UI** — serves `ui/index.html`; calls the API via relative paths (same origin, no CORS) |
| `/api/v1` | **API** — FastAPI + Swagger `/docs` + `/files/{key}` (local audio) |

Two processing modes, chosen automatically by text length (`VIENEU_GPU_MIN_WORDS`, default 600):

- **CPU** (default) — model loaded in-process, synthesis on the VPS. ONNX, torch-free.
- **GPU** (long jobs) — provisions a Vast.ai machine on-demand, ships the voice record + text,
  synthesizes, pulls the WAV back, uploads to storage, then destroys the machine.

Data lives in two places, by role:

- **PostgreSQL** — the "directory": voice records (custom voices: `speaker_emb` + `codes`), and jobs
  (status, progress, and the *key/URL* pointing at the audio). Preset voices stay in RAM, not the DB.
- **Object storage (local disk or Cloudflare R2)** — the "file store": the actual WAV bytes, served
  directly (local via `/files/{key}`, R2 via presigned URL).

The model auto-downloads from Hugging Face on first load (~1.5 GB) into the HF cache.

---

## 🐳 Quick Start — Docker

One `docker-compose.yaml` brings up the whole environment (app + PostgreSQL). **All app config comes
from `.env`** — edit `.env`, re-run `up -d`, and the change takes effect. Nothing is hardcoded in the
compose file.

```bash
cp .env.example .env      # then fill in R2_* / VAST_* if you use the cloud/GPU paths
docker compose up -d
# UI → http://localhost:7862/   ·   API → http://localhost:7862/docs   ·   DB → localhost:7432
```

- Built from [`docker/Dockerfile.vps`](docker/Dockerfile.vps) — `python:3.12-slim` + `uv`, torch-free
  (no `gpu` group).
- Named volume `hf-cache` → `/root/.cache/huggingface` (model isn't re-downloaded each rebuild).
- `./server/data` → local audio when `STORAGE_BACKEND=local`.
- The Postgres container is built from `POSTGRES_*` in `.env`; the app reaches it via `DATABASE_URL`
  (host `db` inside Docker — see the note below).

---

## 🖥️ Run by Hand (no Docker)

Uses [`uv`](https://astral.sh/uv) (not stock pip/venv). You need a PostgreSQL reachable via
`DATABASE_URL`. The simplest option: bring up just the DB from the same compose
(`docker compose up -d db`), which exposes Postgres on `localhost:7432`.

```bash
uv sync                                   # core, torch-free (CPU/ONNX)
# NOTE: running by hand, DATABASE_URL host must be localhost:7432 (not db:5432).
DATABASE_URL=postgresql://vieneu:vieneu@localhost:7432/vieneu \
  uv run python server/main.py           # every other setting comes from .env
```

---

## ⚙️ Configuration

**`.env` at the repo root is the single source of config** (see [`.env.example`](.env.example)). The
compose file hardcodes no app variables, so editing `.env` then `docker compose up -d` is all it takes.

| Var | Default | |
|---|---|---|
| `POSTGRES_USER` / `_PASSWORD` / `_DB` | `vieneu` | credentials the `db` container is built from |
| `DATABASE_URL` | `postgresql://vieneu:vieneu@db:5432/vieneu` | app→DB. Host `db` in Docker; use `localhost:7432` when running by hand |
| `STORAGE_BACKEND` | `local` | `local` (disk, `/files/{key}`) or `r2` (Cloudflare R2 + presigned URL) |
| `VIENEU_SERVE_UI` | `1` | serve the UI at `/`; set `0` to disable |
| `VIENEU_UI_DIR` | `repo-root/ui` | folder holding `index.html` |
| `VIENEU_API_BASE` | *(empty)* | UI→API base; empty = relative/same-origin. Set only if the API is on another host |
| `VIENEU_GPU_MIN_WORDS` | `1000` | auto-route to GPU at/above this word count |
| `MODEL_EAGER_LOAD` | `1` | load the model at startup instead of on first request |
| `R2_*` | — | R2 endpoint / keys / bucket when `STORAGE_BACKEND=r2` |
| `VAST_*` | — | Vast.ai key + GPU job settings (on-demand GPU path) |

> ⚠️ `.env` holds R2 + Vast.ai secrets — it's gitignored. Never commit a real `.env`.

### GPU path — no Docker

The GPU flow runs **on-demand via Vast.ai**: for long text, `server/src/services/gpu_vastai.py` rents
a GPU, Vast pulls the code straight from GitHub and runs it, synthesizes, returns the WAV, and destroys
the machine. There is no Docker image for the GPU tier — the `VAST_*` / `R2_*` vars in `.env` drive it.

---

## 🔌 Public API (`/api/v1`, no token)

| Method | Path | |
|---|---|---|
| GET | `/health` | status + architecture |
| GET | `/styles` | reading styles (`tu_nhien` / `tin_tuc` / `doc_truyen`) |
| GET | `/modes` | processing modes (`cpu` / `gpu`) |
| GET | `/voices` · `/voices/{id}` | list / detail |
| POST | `/voices` | clone a voice from an uploaded clip |
| DELETE | `/voices/{id}` | delete a custom voice |
| POST | `/tts` | create an async job → `{id}` |
| GET | `/tts/{id}` | poll status + progress % |
| DELETE | `/tts/{id}` | cancel |
| GET | `/tts/{id}/download` | download the WAV |

Requests carry an `X-User-Id` header (missing → `default`); a user can only see their own jobs and
custom voices, and the 10 preset voices are shared.

---

## 🔬 Model Overview

| Model | Format | Device | Bilingual | Features | Speed |
|---|---|---|---|---|---|
| **VieNeu-TTS-v3-Turbo** *(this build)* | PyTorch / ONNX | **GPU/CPU** | ✅ | 48 kHz, default voices, cloning, emotion cues, conversation | Fast (batched) |
| **VieNeu-TTS-v2** | PyTorch | GPU | ✅ | Podcast, En-Vi CS | Fast (LMDeploy) |
| **VieNeu-v2-CPU** | GGUF/ONNX | CPU/Edge | ✅ | Podcast, En-Vi CS | Extreme speed |
| **VieNeu-TTS (v1)** | PyTorch | GPU/CPU | ❌ | Stable (Vi only) | Standard |

> On this server, **CPU (ONNX) and GPU produce identical audio quality** (same v3 Turbo, 48 kHz) —
> GPU is only *faster synthesis when warm*, not higher fidelity. With per-job GPU boot overhead, CPU
> in-process is the default for good reason; GPU pays off for long jobs or a kept-warm machine.

---

## 📑 Citation

```bibtex
@misc{vieneutts2026,
  title        = {VieNeu-TTS-v2: Advanced Vietnamese Text-to-Speech with Podcast and Code-Switching Support},
  author       = {Pham Nguyen Ngoc Bao},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/pnnbao-ump/VieNeu-TTS}}
}
```

## 🙏 Acknowledgements

Original model and SDK by **Phạm Nguyễn Ngọc Bảo** ([pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS)).
Uses [MOSS-Audio-Tokenizer-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano)
(v3 Turbo codec) and [sea-g2p](https://github.com/pnnbao97/sea-g2p) for text normalization and
phonemization. License: **Apache 2.0**.

**Made with ❤️ for the Vietnamese TTS community**
