"""VieNeu-TTS REST API (FastAPI).

A thin HTTP layer over the in-process ``Vieneu`` SDK. It exposes the same v3
Turbo synthesis the Gradio UI uses, but as clean REST endpoints under
``/api/v1`` with an auto-generated OpenAPI/Swagger doc at ``/docs``.

The model is loaded ONCE into this process and shared by every request, so the
API and (optionally) the mounted Gradio UI use a single copy of the weights.
"""
