"""VieNeu-TTS model-server (the GPU half of the split PA3 architecture).

Runs the actual v3 Turbo model on the GPU (or CPU for local testing) and exposes
a small HTTP API the CPU backend calls:

- POST /model/v1/jobs         start a synthesis job (returns job id)
- GET  /model/v1/jobs/{id}    poll status + progress %
- DELETE /model/v1/jobs/{id}  cancel
- POST /model/v1/voices       enroll a voice from an uploaded clip
- GET  /files/{key}           download stored audio (local storage backend)

When a job finishes, the audio is uploaded to the configured STORAGE (Cloudflare
R2 in production, or a local folder for dev) and the job exposes the resulting
URL. Requests are authenticated with a shared bearer key.
"""
# Load backend-model/.env before anything reads os.getenv (R2 creds, MODEL_API_KEY).
from . import config as _config  # noqa: F401

__version__ = "1.0.0"
