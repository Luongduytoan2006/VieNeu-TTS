// VieNeu Voice Studio — demo customer frontend. Talks to /api/v1 only.
const API = "/api/v1";
const $ = (id) => document.getElementById(id);

let pollTimer = null;
let currentJob = null;

// ── Tabs ──────────────────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach((t) => {
  t.onclick = () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    $("tab-" + t.dataset.tab).classList.add("active");
    if (t.dataset.tab === "clone") loadVoiceList();
  };
});

// ── Health ────────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    const pill = $("healthPill");
    if (d.status === "ok") {
      pill.textContent = `● sẵn sàng · ${d.backend}/${d.device} · ${d.num_voices} giọng · ${d.active_jobs} job đang chạy`;
      pill.className = "status-pill ok";
    } else {
      pill.textContent = `● ${d.status}${d.model_loaded ? "" : " (đang tải model…)"}`;
      pill.className = "status-pill";
    }
  } catch (e) {
    $("healthPill").textContent = "● mất kết nối API";
    $("healthPill").className = "status-pill err";
  }
}

// ── Load voices + styles ────────────────────────────────────────────────────
async function loadCatalog() {
  try {
    const [vr, sr] = await Promise.all([fetch(`${API}/voices`), fetch(`${API}/styles`)]);
    const voices = await vr.json();
    const styles = await sr.json();
    const vsel = $("voice");
    vsel.innerHTML = "";
    voices.voices.forEach((v) => {
      const o = document.createElement("option");
      o.value = v.id;
      o.textContent = v.label + (v.is_default ? " ⭐" : "");
      if (v.is_default) o.selected = true;
      vsel.appendChild(o);
    });
    const ssel = $("style");
    ssel.innerHTML = "";
    styles.styles.forEach((s) => {
      const o = document.createElement("option");
      o.value = s.id;
      o.textContent = s.label;
      if (s.is_default) o.selected = true;
      ssel.appendChild(o);
    });
  } catch (e) {
    console.error(e);
  }
}

$("temp").oninput = (e) => ($("tempVal").textContent = e.target.value);

// ── Create + poll a TTS job ─────────────────────────────────────────────────
$("genBtn").onclick = async () => {
  const text = $("text").value.trim();
  if (!text) return alert("Nhập văn bản đã!");
  resetResult();
  setBusy(true);

  try {
    const r = await fetch(`${API}/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        voice: $("voice").value || null,
        style: $("style").value,
        temperature: parseFloat($("temp").value),
      }),
    });
    if (!r.ok) {
      const e = await r.json();
      throw new Error(e.detail || r.status);
    }
    const job = await r.json();
    currentJob = job.id;
    $("progWrap").hidden = false;
    $("cancelBtn").disabled = false;
    poll();
  } catch (e) {
    setBusy(false);
    $("progWrap").hidden = false;
    $("progLabel").textContent = "❌ " + e.message;
  }
};

async function poll() {
  if (!currentJob) return;
  try {
    const r = await fetch(`${API}/tts/${currentJob}`);
    const d = await r.json();
    const pct = Math.round(d.progress);
    $("barFill").style.width = pct + "%";
    $("progPct").textContent = pct + "%";
    $("jobMeta").textContent =
      `job ${d.id.slice(0, 8)} · ${d.status} · ${d.done_chunks}/${d.total_chunks} đoạn`;

    if (d.status === "done") {
      $("progLabel").textContent = "✅ Hoàn tất";
      finishJob(d);
      return;
    }
    if (d.status === "cancelled") {
      $("progLabel").textContent = "⏹️ Đã hủy";
      setBusy(false);
      $("cancelBtn").disabled = true;
      return;
    }
    if (d.status === "error") {
      $("progLabel").textContent = "❌ Lỗi: " + (d.error || "");
      setBusy(false);
      $("cancelBtn").disabled = true;
      return;
    }
    $("progLabel").textContent = d.status === "running" ? "Đang tổng hợp…" : "Đang chờ…";
    pollTimer = setTimeout(poll, 1500);
  } catch (e) {
    pollTimer = setTimeout(poll, 2500);
  }
}

function finishJob(d) {
  clearTimeout(pollTimer);
  setBusy(false);
  $("cancelBtn").disabled = true;
  const url = `${API}/tts/${d.id}/download`;
  $("player").src = url;
  $("dlBtn").href = url;
  $("result").hidden = false;
  $("jobMeta").textContent =
    `job ${d.id.slice(0, 8)} · ${d.duration_sec}s audio · xử lý ${d.elapsed_sec}s · ${d.sample_rate}Hz`;
}

$("cancelBtn").onclick = async () => {
  if (!currentJob) return;
  $("cancelBtn").disabled = true;
  await fetch(`${API}/tts/${currentJob}`, { method: "DELETE" });
  // poll() will pick up the cancelled status
};

function setBusy(b) {
  $("genBtn").disabled = b;
  $("orb").classList.toggle("busy", b);
}
function resetResult() {
  clearTimeout(pollTimer);
  $("result").hidden = true;
  $("player").src = "";
  $("barFill").style.width = "0%";
  $("progPct").textContent = "0%";
}

// ── Voice cloning ───────────────────────────────────────────────────────────
$("cloneBtn").onclick = async () => {
  const name = $("vName").value.trim();
  const file = $("vFile").files[0];
  const msg = $("cloneMsg");
  if (!name) return ((msg.className = "clone-msg err"), (msg.textContent = "Nhập tên giọng."));
  if (!file) return ((msg.className = "clone-msg err"), (msg.textContent = "Chọn file audio."));

  msg.className = "clone-msg";
  msg.textContent = "⏳ Đang phân tích giọng…";
  $("cloneBtn").disabled = true;

  const fd = new FormData();
  fd.append("name", name);
  fd.append("audio", file);
  fd.append("description", $("vDesc").value);
  fd.append("gender", $("vGender").value);
  try {
    const r = await fetch(`${API}/voices`, { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.status);
    msg.className = "clone-msg ok";
    msg.textContent = `✅ Đã thêm giọng "${d.id}". Giờ có thể dùng ở tab Tạo giọng nói.`;
    $("vName").value = $("vDesc").value = $("vGender").value = "";
    $("vFile").value = "";
    await loadCatalog();
    await loadVoiceList();
  } catch (e) {
    msg.className = "clone-msg err";
    msg.textContent = "❌ " + e.message;
  } finally {
    $("cloneBtn").disabled = false;
  }
};

async function loadVoiceList() {
  const ul = $("voiceList");
  ul.innerHTML = "<li>…</li>";
  try {
    const d = await (await fetch(`${API}/voices`)).json();
    ul.innerHTML = "";
    d.voices.forEach((v) => {
      const li = document.createElement("li");
      const left = document.createElement("div");
      left.innerHTML =
        `<strong>${v.id}</strong>` +
        `<span class="tag ${v.source}">${v.source}</span>` +
        (v.is_default ? `<span class="tag default">mặc định</span>` : "") +
        `<div class="vmeta">${v.gender || ""} ${v.style}</div>`;
      const del = document.createElement("button");
      del.className = "del";
      del.textContent = "🗑";
      del.title = v.source === "custom" ? "Xóa giọng" : "Không xóa được giọng preset";
      del.disabled = v.source !== "custom";
      del.onclick = async () => {
        if (!confirm(`Xóa giọng "${v.id}"?`)) return;
        await fetch(`${API}/voices/${encodeURIComponent(v.id)}`, { method: "DELETE" });
        loadCatalog();
        loadVoiceList();
      };
      li.appendChild(left);
      li.appendChild(del);
      ul.appendChild(li);
    });
  } catch (e) {
    ul.innerHTML = "<li>lỗi tải danh sách</li>";
  }
}

// ── Boot ────────────────────────────────────────────────────────────────────
checkHealth();
loadCatalog();
setInterval(checkHealth, 5000);
