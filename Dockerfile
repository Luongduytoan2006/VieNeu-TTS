# Same mental model as SoulX-FlashHead/LatentSync/SoulX-LiveAct: modern-glibc base for
# working CUDA runtime + GPU passthrough. VieNeu-TTS manages its OWN Python (3.12,
# required by the lmdeploy GPU wheel pin) via `uv` -- uv installs its own interpreter
# independent of whatever Python ships in the base image (proven during local WSL2
# setup: uv happily installed 3.10 over a 3.8 system Python), and `uv sync --group gpu`
# pip-installs its own torch+cu128 build (which bundles its own CUDA runtime shared
# libraries as nvidia-*-cu12 wheels) -- so the base image's bundled Python/conda/torch
# install is never actually used. `-runtime` (not `-devel`) drops that dead weight while
# still providing the driver-compatible CUDA/cuDNN userspace libs and a matching glibc.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /workspace

# build-essential (gcc): some of uv's synced packages have no prebuilt wheel for this
# platform and compile a plain CPU C extension at install time (confirmed necessary for
# LatentSync's equivalent -runtime build, e.g. insightface/stringzilla) -- nothing
# CUDA-related, so this is far lighter than the full `-devel` CUDA toolchain it replaces.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Full source copied in (not bind-mounted) -- this image is meant to be self-contained
# and publishable; only the HF model cache and generated outputs are volume-mounted
# at runtime (see docker-compose.yml), not the code itself.
COPY . .

RUN uv python install 3.12 && uv sync --group gpu --python 3.12

# vieneu-web binds 127.0.0.1 by default; override via config.yaml or CLI flags at
# runtime if the upstream entrypoint doesn't already read a HOST/PORT env var --
# see entrypoint.sh.
COPY entrypoint.sh /entrypoint.sh
# Strip any Windows CRLF line endings before making it executable: a checkout on
# Windows (autocrlf) turns the shebang into `#!/bin/bash\r`, which the Linux kernel
# can't resolve -> "exec /entrypoint.sh: no such file or directory". sed makes the
# build robust regardless of how the host checked the repo out.
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh
EXPOSE 7862

ENTRYPOINT ["/entrypoint.sh"]
