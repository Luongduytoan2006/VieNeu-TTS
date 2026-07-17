"""voices_repo — CRUD bảng ``voices``. NƠI DUY NHẤT đọc/ghi giọng.

Giọng = danh tính (speaker_emb 192 số + codes) + metadata. Cùng 1 record chạy
cho CẢ CPU (ONNX) lẫn GPU (PyTorch) vì cùng format của SDK tác giả.

2 dạng record trả ra:
* ``get(id)`` / ``list_all()``  → dict "runtime": speaker_emb/codes là numpy array,
  truyền THẲNG vào ``tts.infer(voice=<dict>)`` (cả CPU lẫn GPU-synth_job).
* ``export_json(id)``           → dict JSON-safe (list số) để SCP lên GPU / trả API.

Lý do tách: engine cần numpy để synth; GPU/API cần JSON để truyền qua dây.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

from ..db import connect

logger = logging.getLogger("Vieneu.VoicesRepo")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_runtime(row) -> dict:
    """DB row → dict runtime (numpy) sẵn sàng cho infer(voice=<dict>)."""
    emb = json.loads(row["speaker_emb"]) if row["speaker_emb"] else None
    codes = json.loads(row["codes"]) if row["codes"] else None
    return {
        "id": row["id"],
        "description": row["description"],
        "gender": row["gender"],
        "region": row["region"],
        "style": row["default_style"],          # key 'style' cho _resolve_ref
        "source": row["source"],
        "is_default": bool(row["is_default"]),
        "owner_id": row["owner_id"],
        "speaker_emb": None if emb is None else np.asarray(emb, dtype=np.float32),
        "codes": None if codes is None else np.asarray(codes, dtype=np.int64),
    }


# ── Read ──────────────────────────────────────────────────────────────────────
def get(voice_id: str) -> Optional[dict]:
    """Lấy 1 giọng (runtime dict, numpy) theo id. None nếu không có."""
    with connect() as conn:
        row = conn.execute("SELECT * FROM voices WHERE id = ?", (voice_id,)).fetchone()
    return _row_to_runtime(row) if row else None


def exists(voice_id: str) -> bool:
    with connect() as conn:
        r = conn.execute("SELECT 1 FROM voices WHERE id = ?", (voice_id,)).fetchone()
    return r is not None


def list_all() -> List[dict]:
    """Mọi giọng (runtime dict), preset trước, rồi theo thời gian tạo."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM voices ORDER BY (source='preset') DESC, created_at ASC"
        ).fetchall()
    return [_row_to_runtime(r) for r in rows]


def default_voice_id() -> Optional[str]:
    with connect() as conn:
        row = conn.execute(
            "SELECT id FROM voices WHERE is_default = 1 LIMIT 1"
        ).fetchone()
    return row["id"] if row else None


def export_json(voice_id: str) -> Optional[dict]:
    """Giọng dạng JSON-safe (list số) — để SCP lên GPU hoặc trả ra ngoài."""
    v = get(voice_id)
    if v is None:
        return None
    emb, codes = v["speaker_emb"], v["codes"]
    return {
        "description": v["description"], "gender": v["gender"],
        "style": v["style"],
        "speaker_emb": None if emb is None else [round(float(x), 6) for x in emb.reshape(-1)],
        "codes": None if codes is None else np.asarray(codes, dtype=int).tolist(),
    }


# ── Write ─────────────────────────────────────────────────────────────────────
def upsert(voice_id: str, *, speaker_emb, codes, name: str = "", description: str = "",
           gender: str = "", region: str = "", default_style: str = "tu_nhien",
           source: str = "custom", owner_id: Optional[str] = None,
           ref_audio_key: Optional[str] = None, is_default: bool = False) -> None:
    """Thêm/ghi đè 1 giọng. speaker_emb/codes: numpy array | list | None."""
    emb_j = None if speaker_emb is None else json.dumps(
        [round(float(x), 6) for x in np.asarray(speaker_emb).reshape(-1)])
    codes_j = None if codes is None else json.dumps(np.asarray(codes, dtype=int).tolist())
    now = _now()
    with connect() as conn:
        # Giữ created_at cũ nếu đã tồn tại.
        old = conn.execute("SELECT created_at FROM voices WHERE id = ?", (voice_id,)).fetchone()
        created = old["created_at"] if old else now
        conn.execute(
            """INSERT INTO voices
               (id,name,description,gender,region,default_style,speaker_emb,codes,
                source,owner_id,ref_audio_key,is_default,created_at,updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 name=excluded.name, description=excluded.description,
                 gender=excluded.gender, region=excluded.region,
                 default_style=excluded.default_style, speaker_emb=excluded.speaker_emb,
                 codes=excluded.codes, source=excluded.source, owner_id=excluded.owner_id,
                 ref_audio_key=excluded.ref_audio_key, is_default=excluded.is_default,
                 updated_at=excluded.updated_at""",
            (voice_id, name or voice_id, description, gender, region, default_style,
             emb_j, codes_j, source, owner_id, ref_audio_key,
             1 if is_default else 0, created, now),
        )


def delete(voice_id: str) -> bool:
    with connect() as conn:
        cur = conn.execute("DELETE FROM voices WHERE id = ?", (voice_id,))
    return cur.rowcount > 0


def count() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) c FROM voices").fetchone()["c"]
