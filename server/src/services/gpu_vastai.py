"""GPU backend — provision Vast.ai on-demand: create → run → download → R2 → destroy.

Chạy khi khách chọn GPU mode (context ≥ ngưỡng từ, đã qua 2 lớp check). GPU rất mắc
nên vòng đời: tạo máy → chạy đúng việc → tải WAV về VPS → VPS đẩy R2 → HỦY máy ngay.

Cấu trúc (GRASP — gom endpoint gần nhau vào 1 class):
* ``VastAIClient`` — đóng gói TOÀN BỘ endpoint Vast.ai (init env từ config): verify,
  search, create, poll, ssh/scp, destroy + verify. Mỗi job tạo 1 client riêng.
* ``run(job)``     — hàm ngoài: chạy trọn 1 job GPU (create→...→destroy), try/finally.
* ``get_status(job)`` — hàm ngoài: job đã chạy bao lâu + trạng thái (để stream về UI).

Quirk Vast.ai (đã trả giá bằng tiền thật ở POC):
* search/create/single-instance/DESTROY → /api/v0/ ; CHỈ list → /api/v1/instances/.
* DELETE nhầm v1 → 404 nhưng máy KHÔNG chết → cháy tiền. Luôn destroy v0 + verify lại.
* create trả ``new_contract`` (= instance_id), KHÔNG phải ``id``.
* search: mọi filter (kể cả limit/order) phải nằm trong ``q``; gpu_name có DẤU CÁCH.
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

from ..config import settings

logger = logging.getLogger("Vieneu.GPU")

API_V0 = "https://console.vast.ai/api/v0"
API_V1 = "https://console.vast.ai/api/v1"

# Script chạy trên máy GPU (SCP lên cùng code engine). Nằm cạnh file này.
_HERE = Path(__file__).resolve().parent
SYNTH_JOB = _HERE / "gpu_job" / "synth_job.py"
# Code engine tác giả để SCP lên máy (không phụ thuộc git đã push).
_SERVER = _HERE.parents[1]                       # server/
CODE_DIRS = [_SERVER / "vieneu", _SERVER / "vieneu_utils"]

# Deps runtime tối thiểu cho engine GPU (torch/torchaudio đã có sẵn trong image).
PIP_DEPS = ("sea-g2p>=0.7.14 onnxruntime soundfile soxr tokenizers "
            "huggingface_hub 'transformers>=4.44' neucodec")


class VastAIError(Exception):
    """Lỗi khi thao tác Vast.ai (search/create/ssh/destroy...)."""


class VastAIClient:
    """Bọc REST API Vast.ai cho 1 vòng đời job. Auth bằng ``VAST_AI_API_KEY``.

    Khởi tạo 1 lần/job: đọc toàn bộ cấu hình liên quan từ ``settings`` (env). Giữ
    ``instance_id`` + ``ssh_host/ssh_port`` sau khi tạo để các bước sau dùng lại.
    """

    def __init__(self) -> None:
        self.key = settings.VAST_AI_API_KEY
        if not self.key:
            raise VastAIError("Thiếu VAST_AI_API_KEY trong .env.")
        self.ssh_key = Path(settings.VAST_SSH_KEY).expanduser()
        self.image = settings.VAST_IMAGE
        self.disk_gb = settings.VAST_DISK_GB
        self.poll_sec = settings.VAST_POLL_SEC
        # Trạng thái vòng đời (điền dần).
        self.instance_id: Optional[int] = None
        self.ssh_host: Optional[str] = None
        self.ssh_port: Optional[int] = None
        self.dph: float = 0.0

    # ── HTTP helper ───────────────────────────────────────────────────────────
    def _req(self, method: str, url: str, **kw) -> requests.Response:
        h = kw.pop("headers", {})
        h["Authorization"] = f"Bearer {self.key}"
        return requests.request(method, url, headers=h, timeout=60, **kw)

    # ── Endpoint: account ─────────────────────────────────────────────────────
    def verify(self) -> dict:
        """GET /users/current/ — key sống + số dư. Chặn launch nếu không trả tiền được."""
        # GET /api/v0/users/current/
        # Verify key còn sống + đọc credit/can_pay trước khi thuê máy.
        r = self._req("GET", f"{API_V0}/users/current/")
        if r.status_code != 200:
            raise VastAIError(f"Key Vast.ai lỗi ({r.status_code}): {r.text[:120]}")
        me = r.json()
        if not me.get("can_pay", False):
            raise VastAIError(f"Tài khoản không đủ tiền (credit={me.get('credit')}).")
        return me

    # ── Endpoint: search offer ────────────────────────────────────────────────
    def search_offer(self) -> Optional[dict]:
        """PUT /search/asks/ — tìm GPU rẻ nhất khớp filter. Trả offer rẻ nhất (hoặc None).

        Mọi filter phải nằm trong ``q`` (kể cả order/limit). gpu_name có DẤU CÁCH.
        """
        q = {"q": {
            "rentable": {"eq": True}, "num_gpus": {"eq": 1},
            "gpu_name": {"eq": settings.VAST_GPU_NAME},
            "cuda_max_good": {"gte": settings.VAST_CUDA_MIN},
            "disk_space": {"gte": self.disk_gb},
            "inet_down": {"gte": settings.VAST_MIN_INET_DOWN},
            "dph_total": {"lte": settings.VAST_MAX_DPH},
            "rented": {"eq": False},
            # Ưu tiên máy ĐÁNG TIN (reliability) trước — máy rẻ nhất hay là máy tệ
            # (pull image chậm / không ra được HuggingFace). Lấy nhiều rồi tự xếp.
            "order": [["reliability2", "desc"]], "limit": 20,
        }}
        if settings.VAST_ONLY_VERIFIED:
            q["q"]["verified"] = {"eq": True}
        # PUT /api/v0/search/asks/
        # Tìm danh sách offer GPU khớp filter (giá/cuda/net/verified).
        r = self._req("PUT", f"{API_V0}/search/asks/", json=q)
        if r.status_code != 200:
            raise VastAIError(f"Search lỗi ({r.status_code}): {r.text[:160]}")
        offers = r.json().get("offers", [])
        if not offers:
            return None
        # Xếp hạng: điểm = reliability*1000 + net/100 (net cao pull nhanh) − giá*10.
        def score(o):
            return (o.get("reliability2", 0) * 1000
                    + min(o.get("inet_down", 0), 3000) / 100.0
                    - o.get("dph_total", 1) * 10)
        offers.sort(key=score, reverse=True)
        return offers[0]

    # ── Endpoint: create instance ─────────────────────────────────────────────
    def create(self, offer_id: int) -> int:
        """PUT /asks/{offer_id}/ — thuê máy. Trả instance_id (từ field new_contract).

        ⚠️ TIỀN BẮT ĐẦU TÍNH ngay sau lệnh này (kể cả lúc loading).
        """
        body = {
            "image": self.image, "disk": self.disk_gb,
            "runtype": "ssh_direct", "target_state": "running",
            "onstart": "touch /workspace/.ready",
        }
        # PUT /api/v0/asks/{offer_id}/
        # Thuê máy từ offer đã chọn (image + disk + ssh). ⚠️ TIỀN BẮT ĐẦU TÍNH.
        r = self._req("PUT", f"{API_V0}/asks/{offer_id}/", json=body)
        jr = r.json() if r.content else {}
        iid = jr.get("new_contract")
        if not iid:
            raise VastAIError(f"Tạo instance thất bại: {r.status_code} {jr}")
        self.instance_id = int(iid)
        return self.instance_id

    # ── Endpoint: get single instance ─────────────────────────────────────────
    def get_instance(self, instance_id: Optional[int] = None) -> Optional[dict]:
        """GET /api/v0/instances/{id}/ — chi tiết 1 máy (v1 single trả 404!)."""
        iid = instance_id or self.instance_id
        # GET /api/v0/instances/{id}/
        # Đọc trạng thái 1 máy (actual_status, ssh_host/port, dph). PHẢI dùng v0.
        r = self._req("GET", f"{API_V0}/instances/{iid}/")
        if r.status_code != 200:
            return None
        d = r.json()
        return d.get("instances", d)

    # ── Endpoint: list instances (dùng để verify destroy) ─────────────────────
    def list_instances(self) -> list:
        """GET /api/v1/instances/ — liệt kê máy đang chạy (list DÙNG v1)."""
        # GET /api/v1/instances/
        # Liệt kê mọi máy đang sống — dùng để verify sau destroy (list DÙNG v1).
        r = self._req("GET", f"{API_V1}/instances/")
        if r.status_code != 200:
            return []
        return r.json().get("instances", [])

    # ── Endpoint: destroy (SỐNG CÒN VỀ TIỀN) ──────────────────────────────────
    def destroy(self, instance_id: Optional[int] = None) -> bool:
        """DELETE /api/v0/instances/{id}/ — hủy máy. Trả True nếu đã chết (đã verify).

        Luôn dùng v0 (v1 trả 404 nhưng máy vẫn sống → cháy tiền). Verify lại bằng get.
        """
        iid = instance_id or self.instance_id
        if not iid:
            return True
        # DELETE /api/v0/instances/{id}/
        # HỦY máy để ngừng tính tiền. PHẢI dùng v0 (v1 trả 404 nhưng máy vẫn sống!).
        self._req("DELETE", f"{API_V0}/instances/{iid}/")
        # Verify: get lại phải None/không running.
        inst = self.get_instance(iid)
        dead = inst is None or inst.get("actual_status") in (None, "exited", "offline")
        if not dead:
            # DELETE /api/v0/instances/{id}/  (thử lại 1 lần nếu chưa chết)
            self._req("DELETE", f"{API_V0}/instances/{iid}/")
            inst = self.get_instance(iid)
            dead = inst is None or inst.get("actual_status") in (None, "exited", "offline")
        return dead

    # ── Poll tới RUNNING + lấy ssh host/port ──────────────────────────────────
    def wait_running(self, cancel=None) -> None:
        """Poll (nhịp ``poll_sec``) tới khi actual_status=running + có ssh host/port.

        Fail nhanh nếu máy vào exited/offline/unknown (poll vô hạn = cháy tiền).
        """
        deadline = time.time() + settings.VAST_BOOT_TIMEOUT
        while time.time() < deadline:
            if cancel is not None and cancel.is_set():
                raise VastAIError("Đã hủy khi chờ máy boot.")
            inst = self.get_instance()
            if inst is not None:
                st = inst.get("actual_status")
                host, port = inst.get("ssh_host"), inst.get("ssh_port")
                self.dph = inst.get("dph_total", self.dph) or self.dph
                if st == "running" and host and port:
                    self.ssh_host, self.ssh_port = host, int(port)
                    return
                if st in ("exited", "offline", "unknown"):
                    raise VastAIError(f"Instance vào trạng thái {st} — hủy và thử lại.")
            time.sleep(self.poll_sec)
        raise VastAIError("Timeout chờ máy running.")

    # ── SSH / SCP helpers ─────────────────────────────────────────────────────
    def _ssh_base(self) -> list:
        return ["ssh", "-i", str(self.ssh_key), "-p", str(self.ssh_port),
                "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=10", f"root@{self.ssh_host}"]

    def ssh(self, cmd: str, timeout: int = 600) -> subprocess.CompletedProcess:
        # encoding utf-8 + errors=replace: output SSH có ký tự Việt (RESULT_JSON),
        # nếu để mặc định thì decode bằng cp1252 (Windows) → UnicodeDecodeError.
        return subprocess.run(self._ssh_base() + [cmd], capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=timeout)

    def wait_ssh(self) -> None:
        """Chờ sshd nhận kết nối (ngay sau running vẫn cần vài giây)."""
        for _ in range(30):
            t = self.ssh("echo ok", timeout=15)
            if t.returncode == 0 and "ok" in t.stdout:
                return
            time.sleep(5)
        raise VastAIError("SSH không kết nối được.")

    def scp_up(self, local_paths: list, remote_dir: str = "/workspace/", timeout: int = 180) -> None:
        scp = ["scp", "-i", str(self.ssh_key), "-P", str(self.ssh_port), "-r",
               "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
        args = scp + [str(p) for p in local_paths] + [f"root@{self.ssh_host}:{remote_dir}"]
        subprocess.run(args, check=True, timeout=timeout)

    def scp_down(self, remote_path: str, local_path: Path, timeout: int = 180) -> None:
        scp = ["scp", "-i", str(self.ssh_key), "-P", str(self.ssh_port),
               "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
        local_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(scp + [f"root@{self.ssh_host}:{remote_path}", str(local_path)],
                       check=True, timeout=timeout)

    # ── Bước cao cấp: đưa code lên + chạy synth + tải WAV về ───────────────────
    def setup_and_synth(self, text: str, voice: Optional[str], style: str,
                        out_wav: Path) -> dict:
        """SCP code+script → pip deps → chạy synth_job → tải WAV về ``out_wav``.

        Trả RESULT_JSON (timing + audio_sec + sample_rate). Truyền text qua file để
        tránh giới hạn/escape khi nhét vào lệnh SSH.
        """
        import json

        # 0. HEALTH-CHECK: máy có ra được HuggingFace không? (model tải từ HF).
        # Máy tệ hay không ra được HF → treo ở bước tải model → cháy tiền. Check
        # sớm ~15s: fail thì raise ngay (finally sẽ destroy, chỉ tốn tiền boot).
        hf = self.ssh("curl -s -o /dev/null -w '%{http_code}' --max-time 12 "
                      "https://huggingface.co/api/models/pnnbao-ump/VieNeu-TTS-v3-Turbo",
                      timeout=25)
        if hf.stdout.strip() not in ("200", "301", "302"):
            raise VastAIError(f"Máy không ra được HuggingFace (code={hf.stdout.strip()!r}) "
                              f"— bỏ máy này, thử máy khác.")

        # 1. SCP code engine + synth_job + text lên máy.
        text_local = out_wav.parent / f"{out_wav.stem}.txt"
        text_local.parent.mkdir(parents=True, exist_ok=True)
        text_local.write_text(text, encoding="utf-8")
        self.scp_up(list(CODE_DIRS) + [SYNTH_JOB, text_local])

        # 2. Cài deps tối thiểu (torch có sẵn trong image).
        setup = self.ssh(f"cd /workspace && pip install -q {PIP_DEPS} 2>&1 | tail -3",
                         timeout=900)
        if setup.returncode != 0:
            raise VastAIError(f"pip cài deps lỗi (rc={setup.returncode}): {setup.stdout[-200:]}")

        # 3. Chạy synth_job.py → in RESULT_JSON.
        voice_arg = f'--voice "{voice}"' if voice else ""
        cmd = (f"cd /workspace && PYTHONPATH=/workspace HF_HUB_ENABLE_HF_TRANSFER=0 "
               f'python synth_job.py --out /workspace/out.wav --style {style} '
               f'--text-file /workspace/{text_local.name} {voice_arg}')
        run = self.ssh(cmd, timeout=settings.VAST_JOB_TIMEOUT)
        result = None
        for line in run.stdout.splitlines():
            if line.startswith("RESULT_JSON:"):
                result = json.loads(line[len("RESULT_JSON:"):])
        if not result or not result.get("ok"):
            tail = "\n".join(run.stdout.strip().splitlines()[-8:])
            raise VastAIError(f"synth_job lỗi (rc={run.returncode}): {tail}")

        # 4. Tải WAV về VPS.
        self.scp_down("/workspace/out.wav", out_wav)
        return result


# ── Hàm ngoài: chạy trọn 1 job GPU ────────────────────────────────────────────
def run(job) -> None:
    """Chạy 1 job TTS trên GPU Vast.ai ephemeral (blocking, trong thread của job).

    Vòng đời: create → poll running → SCP+synth → tải WAV về → đẩy R2 → DESTROY.
    Luôn destroy trong ``finally`` (in instance_id sớm để lỡ crash còn hủy tay).
    Tải WAV về VPS rồi VPS đẩy R2 (không đưa R2 secret lên máy community).
    """
    from ..storage import get_storage
    from .jobs import CANCELLED, DONE, ERROR, RUNNING

    t0 = time.time()
    client: Optional[VastAIClient] = None
    out_wav = _SERVER / "data" / "audio" / f"{job.id}.wav"

    try:
        client = VastAIClient()
        job._vast = client                     # để get_status đọc instance_id
        job.status = RUNNING
        job.touch()

        client.verify()
        offer = client.search_offer()
        if not offer:
            raise VastAIError("Không tìm được máy GPU phù hợp (nới VAST_MAX_DPH?).")

        iid = client.create(offer["id"])
        logger.warning("###### VAST INSTANCE_ID=%s (destroy tay: vastai destroy instance %s) ######",
                       iid, iid)
        job.progress = 10.0
        job.touch()

        client.wait_running(cancel=job.cancel)
        if job.cancel.is_set():
            job.status = CANCELLED
            job.touch()
            return
        client.wait_ssh()
        job.progress = 30.0
        job.touch()

        result = client.setup_and_synth(job.text, job.voice, job.style, out_wav)
        if job.cancel.is_set():
            job.status = CANCELLED
            job.touch()
            return
        job.progress = 85.0
        job.touch()

        # Đẩy WAV lên R2 (VPS đẩy — GPU không giữ R2 secret).
        key = f"audio/{job.id}.wav"
        url = get_storage().put(key, out_wav.read_bytes(), "audio/wav")

        job.audio_key = key
        job.audio_url = url
        job.duration_sec = result.get("audio_sec")
        job.sample_rate = result.get("sample_rate")
        job.elapsed_sec = round(time.time() - t0, 3)
        job.progress = 100.0
        job.status = DONE
        job.touch()
        logger.info("✅ GPU job %s xong: %ss audio → %s", job.id[:8], job.duration_sec, url)

    except Exception as e:
        logger.exception("GPU job %s thất bại", job.id[:8])
        job.status = ERROR
        job.error = f"{type(e).__name__}: {e}"
        job.touch()
    finally:
        # DESTROY — luôn chạy, không để cháy tiền GPU.
        if client is not None and client.instance_id is not None:
            try:
                ok = client.destroy()
                logger.info("🗑️ destroy instance %s: %s", client.instance_id,
                            "OK (đã chết)" if ok else "⚠️ VẪN SỐNG — kiểm tra tay!")
            except Exception:
                logger.exception("⚠️ HỦY instance %s thất bại — kiểm tra thủ công!",
                                 client.instance_id)
        # Dọn WAV tạm trên VPS (đã lên R2).
        try:
            out_wav.unlink(missing_ok=True)
            (out_wav.parent / f"{job.id}.txt").unlink(missing_ok=True)
        except Exception:
            pass


# ── Hàm ngoài: trạng thái job GPU (đã chạy bao lâu) ───────────────────────────
def get_status(job) -> dict:
    """Trả trạng thái ngắn gọn của 1 job GPU: chạy bao lâu, instance_id, tiến độ."""
    client = getattr(job, "_vast", None)
    elapsed = round(time.time() - job.created_at.timestamp(), 1)
    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "instance_id": getattr(client, "instance_id", None),
        "dph": getattr(client, "dph", 0.0),
        "elapsed_sec": elapsed,
        "est_cost": round(elapsed / 3600.0 * getattr(client, "dph", 0.0), 5),
    }
