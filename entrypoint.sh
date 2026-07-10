#!/bin/bash
set -e
# Launch the unified server: FastAPI REST API (/api/v1) + Swagger (/docs) + the
# Gradio UI mounted at "/" — ALL on ONE port. This matters for the Proxmox CT,
# which only exposes a single port (7862) through nginx, so UI and API must share it.
#
# Config via env (set in docker-compose.yml):
#   PORT                 -> listen port (default 8000; compose sets 7862)
#   HOST                 -> bind address (default 0.0.0.0)
#   VIENEU_API_MOUNT_UI  -> 1 to mount the Gradio UI at "/" (default 1)
#
# Device is auto-detected inside the app: CUDA -> PyTorch engine, CPU-only -> the
# torch-free ONNX engine. Model weights auto-download from Hugging Face on first
# use into HF_HOME (volume-mounted, so they persist across restarts).
exec uv run vieneu-api
