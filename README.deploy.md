# VieNeu-TTS — split deploy (backend-vps + backend-model)

Two self-contained services. Each has its own `pyproject.toml`, `.env`, and Dockerfile.

```
backend-model/   GPU / AI tier — owns the VieNeu-TTS v3 Turbo model.
                 model_server/ (FastAPI /model/v1, bearer auth) + vieneu SDK (v3turbo)
                 + vieneu_utils. Synthesizes audio, uploads to Cloudflare R2,
                 returns a presigned URL. Listens on :9000.
backend-vps/     CPU / logic tier — torch-free. app/ N-layer FastAPI:
                 config · database · models · schemas · repositories · services · api
                 Public /api/v1 (no auth) + Swagger /docs + demo UI at /. Postgres for
                 durable jobs. Forwards every synthesis to the model-server. Listens on :7862.
docker-compose.yaml   boots BOTH tiers + Postgres on one machine (needs a GPU).
```

Flow: `client → backend-vps /api/v1/tts → backend-model /model/v1/jobs → R2 → URL → backend-vps proxies the WAV back`.

## Public API (`/api/v1`, no auth)

| Method | Path | |
|---|---|---|
| GET | `/health` | readiness + architecture |
| GET | `/styles` | reading styles (tu_nhien / tin_tuc / doc_truyen) |
| GET | `/modes` | synthesis modes (v3turbo / pytorch / onnx) |
| GET | `/voices` · `/voices/{id}` | list / detail |
| POST | `/voices` | clone a voice from an uploaded clip |
| DELETE | `/voices/{id}` | remove a custom voice |
| POST | `/tts` | create async job → `{id}` |
| GET | `/tts/{id}` | poll status + progress % |
| DELETE | `/tts/{id}` | cancel |
| GET | `/tts/{id}/download` | download the WAV |

## Run — both tiers on one GPU box (dev / all-in-one)

```bash
cp backend-model/.env.example backend-model/.env   # fill R2 + MODEL_API_KEY
cp backend-vps/.env.example  backend-vps/.env       # set MODEL_API_KEY (same value)
docker compose up -d --build
# API:  http://localhost:7862/docs   ·   Model: http://localhost:9000/model/v1/health
```

## Run — split (production)

**GPU box** (has the model, exposed via a tunnel on :9000):
```bash
cd backend-model
cp .env.example .env      # STORAGE_BACKEND=r2 + R2_* + MODEL_API_KEY
docker build -f Dockerfile.gpu -t vieneu-model:gpu .
docker run -d --gpus all -p 9000:9000 --env-file .env vieneu-model:gpu
# tunnel a public hostname → :9000  (e.g. vienue-tts.luongduytoan.io.vn)
```

**CPU VPS** (no model — just uv, torch-free):
```bash
cd backend-vps
cp .env.example .env
#  MODEL_SERVER_URL = the GPU tunnel URL (https://vienue-tts.luongduytoan.io.vn)
#  MODEL_API_KEY    = same secret as the GPU box
#  VIENEU_DATABASE_URL = your Postgres
uv sync
uv run python -m app.main       # serves :7862, /api/v1 + /docs
```

## Secrets

`*.env` files are gitignored (they hold R2 keys + the shared `MODEL_API_KEY`). Only the
`.env.example` templates are committed. Never commit a real `.env`.
