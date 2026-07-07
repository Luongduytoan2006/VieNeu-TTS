# VieNeu-TTS — Run Guide (from a live smoke test)

Self-hosted **Vietnamese-native bilingual (VN/EN) TTS**. 10,000+ hrs training data, instant voice
cloning (3–5 s reference), experimental emotion / non-verbal tags (`[cười]` laugh, `[thở dài]` sigh).
Apache-2.0. GPU server path (LMDeploy) **and** a torch-free ONNX **CPU** path. This is the
**recommended TTS pick** for the AI-live-streaming POC (Vietnamese-native, free to self-host, small
enough to co-locate on the avatar GPU).

> **✅ Works out of the box — no code changes needed.** Verified end-to-end on a rented RTX 3060.

---

## ✅ TL;DR — run it

Native install (uses **`uv`**, not stock pip/venv — `uv venv` doesn't bootstrap pip):

```bash
uv sync --group gpu          # torch+cu128 + all deps (~1.5 min cold). CPU-only: use the CPU group.
uv run vieneu-web            # launches the Gradio web app (~2.5 min to serving)
```

Then in the UI (or via the Gradio API):
- **Backbone:** `VieNeu-TTS-v3-Turbo` · **Codec:** `VieNeu-Codec` · **Device:** `CUDA`
- **lmdeploy optimization:** enabled (`force_lmdeploy=True`) — big speedup on GPU
- pick a preset voice (e.g. `Mai Anh`) or clone one from 3–5 s of reference audio
- Weights **auto-download from Hugging Face on first `/load_model`** (~34 s) — no manual download step.

---

## 📊 Measured performance & cost (RTX 3060 12 GB, Vast.ai)

| Stage | Time |
|---|---|
| `uv sync --group gpu` (cold) | ~1 min 35 s |
| App startup → Gradio serving | ~2 min 34 s |
| Model load (`/load_model`) | ~34 s |
| **Total cold start** | **~4.7 min** |
| Generation #1 (cold) | 4.92 s synth / **0.80× real-time** |
| Generation #2 (warm) | 3.20 s synth / **1.30× real-time** |

- **Faster than real-time once warm (1.30×).** First call is slower (lmdeploy/CUDA kernel warm-up) —
  a non-issue for a long-running service.
- Cost on RTX 3060: **~$0.06/hr** (Vast) — trivial; runs comfortably alongside the avatar model.

---

## 🐛 Gotchas

1. **Use `uv`, not `python -m venv`/pip.** `uv venv` doesn't install pip; use `uv pip install` for any
   manual additions. `uv sync --group gpu` sets up the CUDA build.
2. **First generation is slower than real-time (0.80×)** — lmdeploy/CUDA warm-up on the first call
   only; steady state is 1.30× real-time.
3. **Emotion/non-verbal tags are experimental** (`[cười]`, `[thở dài]`, etc.) — quality varies.
4. CPU-only path exists (torch-free ONNX) for hosts without a GPU — much slower but zero GPU cost.

---

## 💡 POC notes

- **Recommended TTS for the pipeline.** Vietnamese-native (not a foreign model with unverified
  transfer), Apache-2.0, self-hostable on the existing GPU budget. Cloud version at **vieneu.io** is
  also the cheapest hosted option found (see project `PRICING.md`).
- Chunk-based TTS from a script (POC goal #1) fits naturally: warm the model once, then each chunk
  synthesizes at ~1.3× real-time.
- Verdict from smoke test: **works as-is, nothing to modify.**
