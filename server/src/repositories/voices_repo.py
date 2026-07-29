"""voices_repo — CRUD bảng ``voices`` (giọng CUSTOM của user). NƠI DUY NHẤT đọc/ghi.

Chỉ chứa giọng do user clone; preset nằm ở RAM engine (không vào DB). Mọi hàm nhận
``user_ref`` → mỗi người 1 kho riêng, khóa (user_ref, id) nên 2 user được trùng tên.

speaker_emb/codes lưu cột JSONB → psycopg tự serialize list<->JSON (dùng ``Jsonb``).
2 dạng record trả ra:
* ``get`` / ``list_for_user``  → dict "runtime": speaker_emb/codes numpy, truyền
  THẲNG vào ``tts.infer(voice=<dict>)`` (CPU lẫn GPU-synth_job).
* ``export_json``              → dict JSON-safe (list số) để SCP lên GPU / trả API.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from psycopg.types.json import Jsonb

from ..db import connect

logger = logging.getLogger("Vieneu.VoicesRepo")


def _row_to_runtime(row) -> dict:
    """DB row → dict runtime (numpy) sẵn sàng cho infer(voice=<dict>). JSONB đã là
    list Python sẵn (psycopg tự parse), chỉ cần bọc numpy."""
    emb, codes = row["speaker_emb"], row["codes"]
    return {
        "id": row["id"],
        "user_ref": row["user_ref"],
        "description": row["description"],
        "gender": row["gender"],
        "region": row["region"],
        "style": row["default_style"],          # key 'style' cho _resolve_ref
        "source": "custom",
        "is_default": False,
        "speaker_emb": None if emb is None else np.asarray(emb, dtype=np.float32),
        "codes": None if codes is None else np.asarray(codes, dtype=np.int64),
    }


# ── Read ──────────────────────────────────────────────────────────────────────
def get(user_ref: str, voice_id: str) -> Optional[dict]:
    """Lấy 1 giọng custom (runtime dict) của user. None nếu không có."""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM voices WHERE user_ref = %s AND id = %s", (user_ref, voice_id)
        ).fetchone()
    return _row_to_runtime(row) if row else None


def exists(user_ref: str, voice_id: str) -> bool:
    with connect() as conn:
        r = conn.execute(
            "SELECT 1 FROM voices WHERE user_ref = %s AND id = %s", (user_ref, voice_id)
        ).fetchone()
    return r is not None


def list_for_user(user_ref: str) -> List[dict]:
    """Mọi giọng custom của 1 user (runtime dict), cũ→mới."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM voices WHERE user_ref = %s ORDER BY created_at ASC", (user_ref,)
        ).fetchall()
    return [_row_to_runtime(r) for r in rows]


def export_json(user_ref: str, voice_id: str) -> Optional[dict]:
    """Giọng dạng JSON-safe (list số) — để SCP lên GPU hoặc trả ra ngoài."""
    v = get(user_ref, voice_id)
    return _to_json(v) if v is not None else None


def _to_json(v: dict) -> dict:
    emb, codes = v.get("speaker_emb"), v.get("codes")
    return {
        "description": v.get("description", ""), "gender": v.get("gender", ""),
        "style": v.get("style", "tu_nhien"),
        "speaker_emb": None if emb is None else [round(float(x), 6) for x in np.asarray(emb).reshape(-1)],
        "codes": None if codes is None else np.asarray(codes, dtype=int).tolist(),
    }


# ── Write ─────────────────────────────────────────────────────────────────────
def upsert(user_ref: str, voice_id: str, *, speaker_emb, codes, name: str = "",
           description: str = "", gender: str = "", region: str = "",
           default_style: str = "tu_nhien", ref_audio_key: Optional[str] = None) -> None:
    """Thêm/ghi đè 1 giọng custom của user. speaker_emb/codes: numpy|list|None."""
    emb_j = None if speaker_emb is None else Jsonb(
        [round(float(x), 6) for x in np.asarray(speaker_emb).reshape(-1)])
    codes_j = None if codes is None else Jsonb(np.asarray(codes, dtype=int).tolist())
    with connect() as conn:
        conn.execute(
            """INSERT INTO voices
               (user_ref,id,name,description,gender,region,default_style,
                speaker_emb,codes,ref_audio_key,updated_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
               ON CONFLICT(user_ref,id) DO UPDATE SET
                 name=excluded.name, description=excluded.description,
                 gender=excluded.gender, region=excluded.region,
                 default_style=excluded.default_style, speaker_emb=excluded.speaker_emb,
                 codes=excluded.codes, ref_audio_key=excluded.ref_audio_key,
                 updated_at=now()""",
            (user_ref, voice_id, name or voice_id, description, gender, region,
             default_style, emb_j, codes_j, ref_audio_key),
        )


def delete(user_ref: str, voice_id: str) -> bool:
    with connect() as conn:
        cur = conn.execute(
            "DELETE FROM voices WHERE user_ref = %s AND id = %s", (user_ref, voice_id))
    return cur.rowcount > 0


def count(user_ref: Optional[str] = None) -> int:
    with connect() as conn:
        if user_ref is None:
            return conn.execute("SELECT COUNT(*) c FROM voices").fetchone()["c"]
        return conn.execute(
            "SELECT COUNT(*) c FROM voices WHERE user_ref = %s", (user_ref,)
        ).fetchone()["c"]
