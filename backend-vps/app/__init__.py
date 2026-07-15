"""VieNeu-TTS backend-vps — the CPU/logic tier.

An N-layer FastAPI service (api → services → repositories → database) that owns the
public REST API (/api/v1) and job durability (Postgres). It holds NO model: every
synthesis is forwarded over HTTP to the GPU model-server (backend-model).
"""
