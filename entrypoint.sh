#!/bin/bash
set -e
# vieneu-web itself already reads GRADIO_SERVER_NAME/GRADIO_SERVER_PORT (see
# apps/gradio_main.py) and auto-disables the public share link when bound to
# 0.0.0.0 -- both are set via docker-compose.yml's environment section, not here.
# Model weights auto-download from Hugging Face on first use (the vieneu SDK's own
# behavior, not something this entrypoint needs to orchestrate) into HF_HOME, which
# is volume-mounted so they persist across container restarts.
exec uv run vieneu-web
