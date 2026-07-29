"""src — backend chính cho VieNeu-TTS (1 server, 2 mode CPU/GPU).

Bọc code gốc tác giả (``vieneu`` / ``vieneu_utils`` — 'đầu IO') và xuất ra 1
Swagger duy nhất ở ``/api/v1``. Kiến trúc if/else theo mode:

* **CPU mode** (mặc định) — model nạp sẵn in-process, synth ngay trên VPS.
* **GPU mode** (nâng cao, context > 1000 ký tự) — provision Vast.ai on-demand:
  tạo máy → chạy → upload R2 → destroy. (Tạm để skeleton, bật sau.)
"""

__all__ = ["config", "engine", "storage", "schemas"]
