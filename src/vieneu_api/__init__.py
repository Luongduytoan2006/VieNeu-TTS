"""VieNeu-TTS async job-based REST backend.

A production-shaped FastAPI service over the in-process ``Vieneu`` v3 Turbo SDK:

- Async TTS jobs (uuid): create → poll % progress → cancel → download the WAV.
- Voice CRUD (clone a voice from an uploaded audio clip).
- Catalog (voices / styles / modes) and health.

The model is loaded ONCE at startup and kept hot, so requests never wait on a
cold load. Job/voice metadata lives in Postgres; audio files live on disk keyed
by uuid, served via a download endpoint.
"""

__all__ = ["__version__"]
__version__ = "1.0.0"
