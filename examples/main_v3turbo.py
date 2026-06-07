"""
VieNeu-TTS v3 Turbo — minimal SDK inference example.

Run:
    uv run python examples/main_v3turbo.py

Requires the `vieneu` package (pip install -e . from the repo root, or have
`src/` on the path — the bootstrap below handles the latter automatically).
"""
import sys
from pathlib import Path

# Bootstrap: allow running straight from the repo without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vieneu import Vieneu


def main() -> None:
    # Loads the v3 Turbo model + the 16 built-in default voices (48 kHz).
    tts = Vieneu(mode="v3turbo")

    # List the built-in preset voices.
    print("Available preset voices:")
    for name, desc in tts.list_preset_voices():
        print(f"  - {name}: {desc}")

    text = ("Xin chào, đây là VieNeu-TTS phiên bản ba Turbo. "
            "Hôm nay trời đẹp, mời bạn cùng nghe thử giọng đọc tiếng Việt.")

    # Synthesize with the "Xuân Vĩnh" default voice. Default voices use the
    # speaker reserved token + fixed reference codes ("emotion" path) -> fast and
    # stable, no reference encoding needed.
    wav = tts.infer(text, voice="Xuân Vĩnh")
    out = "xuanvinh_output.wav"
    tts.save(wav, out)
    print(f"\nSaved {out}  ({len(wav) / tts.sample_rate:.2f}s @ {tts.sample_rate} Hz)")

    # ── Other usage patterns ────────────────────────────────────────────────
    # wav = tts.infer(text)                                 # default voice (Ngọc Lan)
    # wav = tts.infer(text, voice="Ngọc Linh")              # another default voice
    # wav = tts.infer(text, ref_audio="my_voice.wav")       # clone your own voice
    # for chunk in tts.infer_stream(text, voice="Xuân Vĩnh"): # low-latency streaming
    #     ...  # play / send each chunk as it is produced


if __name__ == "__main__":
    main()
