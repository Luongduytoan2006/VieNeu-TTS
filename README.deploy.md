# VieNeu-TTS — deploy

Kiến trúc hiện tại: **1 server** chạy toàn bộ app (torch-free, CPU/ONNX in-process).
`server/main.py` bật cùng lúc 2 cổng:

| Cổng | |
|---|---|
| `7862` | API FastAPI — prefix `/api/v1` + Swagger `/docs` + `/files/{key}` (audio local) |
| `7870` | Giao diện demo — phục vụ `demo/index.html`, gọi ngược lại API :7862 |

Model VieNeu-TTS v3 Turbo tự tải từ Hugging Face lần đầu (~1.5GB) vào cache HF.

## Deploy VPS CPU bằng Docker (khuyến nghị)

Chạy đúng một lệnh ở repo root:

```bash
docker compose up -d
# API:  http://localhost:7862/docs      ·      Demo: http://localhost:7870
```

- Build từ [`docker/Dockerfile.vps`](docker/Dockerfile.vps) (base `python:3.12-slim` + `uv`,
  `uv sync` torch-free — KHÔNG cài group `gpu`).
- `docker-compose.yaml` map cổng `7862` + `7870`, nạp `.env` ở repo root, mount:
  - named volume `hf-cache` → `/root/.cache/huggingface` (không tải lại model mỗi lần dựng),
  - `./server/data` → audio local (khi `STORAGE_BACKEND=local`).
- `MODEL_EAGER_LOAD=1`: nạp model ngay lúc khởi động. Lần đầu chờ tải HF (~1.5GB); các
  lần sau nhanh nhờ volume cache.

Cấu hình qua `.env` (repo root) hoặc `environment:` trong compose. Mặc định
`STORAGE_BACKEND=local` (audio lưu đĩa, phục vụ qua `/files/{key}`). Muốn dùng Cloudflare
R2: đặt `STORAGE_BACKEND=r2` + các biến `R2_*`.

## Chạy tay (không Docker)

```bash
uv sync                                    # core torch-free (CPU)
MODEL_EAGER_LOAD=1 uv run python server/main.py
```

## Server GPU — KHÔNG dùng Docker

Luồng GPU chạy **on-demand qua Vast.ai**: khi context đủ dài (`VIENEU_GPU_MIN_WORDS`),
`server/src/services/gpu_vastai.py` thuê máy GPU, Vast.ai kéo code thẳng từ GitHub rồi tự
chạy, synth xong lấy WAV về và tắt máy. Không có Docker image cho tầng GPU. Các biến
`VAST_*` / `R2_*` trong `.env` phục vụ luồng này.

## Public API (`/api/v1`, không cần token)

| Method | Path | |
|---|---|---|
| GET | `/health` | trạng thái + kiến trúc |
| GET | `/styles` | phong cách đọc (tu_nhien / tin_tuc / doc_truyen) |
| GET | `/modes` | chế độ xử lý (cpu / gpu) |
| GET | `/voices` · `/voices/{id}` | danh sách / chi tiết giọng |
| POST | `/voices` | clone giọng từ clip tải lên |
| DELETE | `/voices/{id}` | xóa giọng tùy chỉnh |
| POST | `/tts` | tạo job bất đồng bộ → `{id}` |
| GET | `/tts/{id}` | poll trạng thái + % tiến độ |
| DELETE | `/tts/{id}` | hủy |
| GET | `/tts/{id}/download` | tải WAV |

## Bảo mật

`.env` chứa khóa R2 + Vast.ai — đã gitignore, KHÔNG commit `.env` thật.
