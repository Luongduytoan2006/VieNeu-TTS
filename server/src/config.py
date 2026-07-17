"""Runtime settings cho server/src, override được qua biến môi trường.

Nạp 1 lần khi import. `.env` NẰM Ở REPO ROOT (nguồn chính: R2, Vast.ai, model…);
``server/.env`` chỉ là fallback tùy chọn. Giá trị đã có trong môi trường thật luôn
thắng (``setdefault``), nên Docker ``environment:`` vẫn override được.
"""
from __future__ import annotations

import os
from pathlib import Path

# server/  (một cấp trên src/)
_ROOT = Path(__file__).resolve().parents[1]
# repo root (chứa .env chính + pyproject.toml + uv.lock)
_REPO_ROOT = _ROOT.parent


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


# repo-root/.env là nguồn chính (load trước → ưu tiên); server/.env chỉ fallback.
_load_dotenv(_REPO_ROOT / ".env")
_load_dotenv(_ROOT / ".env")

# Thư mục dữ liệu cục bộ (audio CPU + uploads) — tạo sẵn.
(_ROOT / "data").mkdir(parents=True, exist_ok=True)


def _as_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


class Settings:
    # ── HTTP server (public API) ─────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "7862"))
    API_PREFIX: str = "/api/v1"
    FORCE_HTTPS: bool = _as_bool(os.getenv("VIENEU_FORCE_HTTPS", "0"))
    # CORS: giao diện chạy ở cổng khác nên trình duyệt cần API cho phép cross-origin.
    # Mặc định "*" (mở) — thu hẹp bằng VIENEU_CORS_ORIGINS="http://a,http://b" khi prod.
    CORS_ORIGINS: str = os.getenv("VIENEU_CORS_ORIGINS", "*")

    # ── Model (CPU/in-process) ───────────────────────────────────────────────
    BACKBONE_REPO: str = os.getenv("VIENEU_BACKBONE_REPO", "pnnbao-ump/VieNeu-TTS-v3-Turbo")
    # device=auto → CUDA (PyTorch) nếu có, else ONNX/CPU. Server này mặc định CPU.
    MODEL_DEVICE: str = os.getenv("VIENEU_DEVICE", "auto")
    MODEL_EAGER_LOAD: bool = _as_bool(os.getenv("MODEL_EAGER_LOAD", "1"))
    MAX_CHARS_PER_CHUNK: int = int(os.getenv("VIENEU_MAX_CHARS", "256"))
    MAX_NEW_FRAMES: int = int(os.getenv("VIENEU_MAX_NEW_FRAMES", "300"))

    # ── Ngưỡng chọn mode ─────────────────────────────────────────────────────
    # GPU chỉ được phép khi context đủ dài (đếm SỐ TỪ, tách theo khoảng trắng).
    # FE đã chặn 1 lớp; BE chặn thêm lớp nữa ở services/createVoice.py.
    GPU_MIN_WORDS: int = int(os.getenv("VIENEU_GPU_MIN_WORDS", "1000"))

    # ── Storage (audio kết quả) ──────────────────────────────────────────────
    # local  → lưu đĩa, phục vụ qua /files/{key} (CPU dev/single-machine).
    # r2     → Cloudflare R2 (S3), trả presigned URL (mặc định khi GPU/prod).
    STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "local").lower()
    # Key audio đã có prefix "audio/" → trỏ thẳng vào data/ để tránh lặp audio/audio.
    STORAGE_LOCAL_DIR: str = os.getenv("STORAGE_LOCAL_DIR", str(_ROOT / "data"))
    PUBLIC_BASE_URL: str = os.getenv("VIENEU_PUBLIC_BASE", f"http://localhost:{os.getenv('PORT', '7862')}")

    # R2 (đọc từ repo-root .env — hỗ trợ cả tên biến hoa/thường trong .env hiện có)
    R2_URL: str = os.getenv("R2_URL") or os.getenv("R2_url", "")
    R2_ACCESS_KEY: str = os.getenv("R2_ACCESS_KEY") or os.getenv("R2_access_key", "")
    R2_SECRET_KEY: str = os.getenv("R2_SECRET_KEY") or os.getenv("R2_secret_key", "")
    R2_BUCKET: str = os.getenv("R2_BUCKET", "vieneu-tts")
    R2_PUBLIC_BASE: str = os.getenv("R2_PUBLIC_BASE", "")
    R2_PRESIGN_TTL: int = int(os.getenv("R2_PRESIGN_TTL", "86400"))

    # ── Vast.ai (GPU on-demand) — dùng ở services/gpu_vastai.py ──────────────
    VAST_AI_API_KEY: str = os.getenv("VAST_AI_API_KEY", "")
    VAST_GPU_NAME: str = os.getenv("VAST_GPU_NAME", "RTX 3060")     # tên có DẤU CÁCH
    VAST_MAX_DPH: float = float(os.getenv("VAST_MAX_DPH", "0.10"))  # giá tối đa $/giờ
    VAST_CUDA_MIN: float = float(os.getenv("VAST_CUDA_MIN", "12.8"))  # yêu cầu tác giả
    VAST_DISK_GB: int = int(os.getenv("VAST_DISK_GB", "40"))
    VAST_MIN_INET_DOWN: int = int(os.getenv("VAST_MIN_INET_DOWN", "300"))  # Mbps
    VAST_ONLY_VERIFIED: bool = _as_bool(os.getenv("VAST_ONLY_VERIFIED", "1"))
    VAST_IMAGE: str = os.getenv("VAST_IMAGE", "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel")
    # SSH private key để SCP code + lấy WAV về (đã add public key lên Vast account).
    VAST_SSH_KEY: str = os.getenv("VAST_SSH_KEY", str(Path.home() / ".ssh" / "Desktoplenovo"))
    VAST_BOOT_TIMEOUT: int = int(os.getenv("VAST_BOOT_TIMEOUT", "600"))   # chờ running (s)
    VAST_JOB_TIMEOUT: int = int(os.getenv("VAST_JOB_TIMEOUT", "1200"))    # chờ synth (s)
    VAST_POLL_SEC: int = int(os.getenv("VAST_POLL_SEC", "30"))            # nhịp poll (tiền thật!)

    # ── Giao diện demo (tách ngoài server/, chạy CỔNG RIÊNG) ─────────────────
    # main.py phục vụ demo/index.html ở cổng này (khác cổng API). Tắt: =0.
    SERVE_DEMO: bool = _as_bool(os.getenv("VIENEU_SERVE_DEMO", "1"))
    DEMO_PORT: int = int(os.getenv("VIENEU_DEMO_PORT", "7870"))
    # Thư mục chứa giao diện — mặc định repo-root/demo (ngoài server/).
    DEMO_DIR: str = os.getenv("VIENEU_DEMO_DIR", str(_REPO_ROOT / "demo"))
    # URL API mà giao diện gọi tới (client-side). Mặc định localhost:PORT.
    API_BASE_URL: str = os.getenv("VIENEU_API_BASE", f"http://localhost:{PORT}")


settings = Settings()
