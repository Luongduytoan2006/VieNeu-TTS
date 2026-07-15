"""HTTP client for the GPU model-server (backend-model).

Wraps the model-server's ``/model/v1`` API with the shared bearer key. This is the
ONLY module in backend-vps that speaks to the model-server, so the split is easy to
reason about: everything model-related is one HTTP hop behind this class.
"""
from __future__ import annotations

import logging
from typing import Optional

import requests

from ..config import settings

logger = logging.getLogger("Vieneu.VPS.model_client")


class ModelClientError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(f"[{status}] {detail}")
        self.status = status
        self.detail = detail


def _detail(r: requests.Response) -> str:
    try:
        return r.json().get("detail", r.text)
    except Exception:
        return r.text


class ModelClient:
    def __init__(self) -> None:
        self.base = settings.MODEL_SERVER_URL.rstrip("/")
        self.prefix = f"{self.base}/model/v1"
        self._headers = {"Authorization": f"Bearer {settings.MODEL_API_KEY}"}

    # ── health / catalog ─────────────────────────────────────────────────────
    def health(self) -> dict:
        r = requests.get(f"{self.prefix}/health", timeout=8)
        r.raise_for_status()
        return r.json()

    def voices(self) -> dict:
        r = requests.get(f"{self.prefix}/voices", headers=self._headers, timeout=15)
        r.raise_for_status()
        return r.json()

    def enroll_voice(self, name: str, audio_bytes: bytes, filename: str, *,
                     description="", gender="", style="tu_nhien", denoise=True) -> dict:
        files = {"audio": (filename, audio_bytes, "audio/wav")}
        data = {"name": name, "description": description, "gender": gender,
                "style": style, "denoise": str(denoise).lower()}
        r = requests.post(f"{self.prefix}/voices", headers=self._headers,
                          files=files, data=data, timeout=120)
        if r.status_code >= 400:
            raise ModelClientError(r.status_code, _detail(r))
        return r.json()

    def delete_voice(self, voice_id: str) -> None:
        r = requests.delete(f"{self.prefix}/voices/{voice_id}", headers=self._headers, timeout=15)
        if r.status_code not in (204, 404):
            raise ModelClientError(r.status_code, _detail(r))

    # ── jobs ─────────────────────────────────────────────────────────────────
    def create_job(self, text: str, voice: Optional[str], style: str,
                   temperature: float, max_chars: int) -> dict:
        r = requests.post(f"{self.prefix}/jobs", headers=self._headers, timeout=15, json={
            "text": text, "voice": voice, "style": style,
            "temperature": temperature, "max_chars": max_chars})
        if r.status_code >= 400:
            raise ModelClientError(r.status_code, _detail(r))
        return r.json()

    def get_job(self, remote_id: str) -> dict:
        r = requests.get(f"{self.prefix}/jobs/{remote_id}", headers=self._headers, timeout=10)
        if r.status_code >= 400:
            raise ModelClientError(r.status_code, _detail(r))
        return r.json()

    def cancel_job(self, remote_id: str) -> dict:
        r = requests.delete(f"{self.prefix}/jobs/{remote_id}", headers=self._headers, timeout=10)
        if r.status_code >= 400:
            raise ModelClientError(r.status_code, _detail(r))
        return r.json()

    def fetch_audio(self, url: str) -> bytes:
        """Download finished audio from the model-server's storage URL (presigned R2
        or the model-server's /files route). No creds needed — the URL carries them."""
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return r.content


client = ModelClient()
