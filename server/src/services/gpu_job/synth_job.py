"""synth_job.py — chạy TRÊN máy GPU Vast.ai.

Nhận 1 đoạn text, nạp engine VieNeu v3 Turbo (PyTorch/CUDA), sinh audio THEO TỪNG
CHUNK (giống đường CPU in-process) để BÁO TIẾN ĐỘ real-time, lưu WAV, và in TIMING
từng bước ra stdout dạng JSON dòng cuối (prefix ``RESULT_JSON:``).

Tiến độ real-time (orchestrator ở VPS đọc stdout live để cập nhật job.progress):
* ``PROGRESS:phase=load_model``            — bắt đầu nạp model (tải HF lần đầu).
* ``PROGRESS:phase=synth total=N``         — model xong, bắt đầu synth N chunk.
* ``PROGRESS:chunk=i total=N``             — vừa synth xong chunk thứ i.
Mỗi dòng PROGRESS được flush ngay để VPS nhận tức thì (không đợi buffer).

Chạy:  python synth_job.py --out /workspace/out.wav [--text-file f.txt] [--voice "Ngọc Lan"]

Chỉ phụ thuộc engine tác giả (``vieneu`` + ``vieneu_utils``) — KHÔNG import backend
``src`` đang refactor. Watermark tắt mặc định (khỏi cài ``perth``).
"""
import argparse
import json
import sys
import time

# 360 từ tiếng Việt (đếm bằng khoảng trắng) — đủ dài để đo throughput ổn định.
DEFAULT_TEXT = (
    "Hôm nay trời thật đẹp và trong xanh, nắng vàng rực rỡ chan hòa khắp phố phường. "
    "Mọi người ai nấy đều vui vẻ ra đường dạo chơi, tận hưởng không khí mát mẻ dễ chịu "
    "của một buổi sáng cuối tuần yên bình bên gia đình và bạn bè thân thương gần gũi. "
    "Chúng ta cùng nhau trò chuyện, ăn uống, vui đùa thỏa thích suốt cả một ngày dài, "
    "rồi lại mong chờ đến những dịp nghỉ ngơi thư giãn tiếp theo để có thêm thật nhiều "
    "kỷ niệm đẹp đẽ khó quên trong cuộc đời. Mỗi con người chúng ta luôn trân trọng "
    "từng khoảnh khắc quý giá bên nhau, bởi thời gian trôi qua rất nhanh và không bao giờ "
    "quay trở lại. Vì thế, hãy sống thật trọn vẹn, yêu thương nhiều hơn, và biết ơn những "
    "điều bình dị nhất mà cuộc sống mang lại mỗi ngày. Khi bình minh lên, chim hót líu lo "
    "trên những tán cây xanh mướt, gió nhẹ thổi qua làm lay động những nhành hoa đang khoe "
    "sắc thắm. Trẻ em nô đùa chạy nhảy trên con đường làng quen thuộc, tiếng cười giòn tan "
    "vang vọng khắp không gian tĩnh lặng của buổi sớm mai. Người nông dân ra đồng từ sớm, "
    "chăm chỉ cày cấy vun trồng cho mùa màng bội thu, gương mặt rạng rỡ niềm tin vào một "
    "tương lai tươi sáng. Cuộc sống dù còn nhiều khó khăn vất vả, nhưng chỉ cần chúng ta "
    "luôn giữ vững niềm tin, lạc quan và cố gắng không ngừng, thì mọi ước mơ rồi sẽ thành "
    "hiện thực. Hãy cùng nhau xây dựng một cộng đồng đoàn kết, yêu thương và sẻ chia, để "
    "quê hương ngày càng giàu đẹp, văn minh và đáng sống hơn cho tất cả mọi người. Mỗi sáng "
    "thức dậy là một cơ hội mới để bắt đầu lại, để làm những điều tốt đẹp và lan tỏa yêu "
    "thương đến khắp muôn nơi trên dải đất hình chữ S thân yêu này của chúng ta."
)


def _emit(msg: str) -> None:
    """In 1 dòng PROGRESS + flush ngay (để VPS đọc stdout live nhận tức thì)."""
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/out.wav")
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--voice", default=None, help="ten voice preset; bo trong = default")
    ap.add_argument("--voice-file", default=None,
                    help="JSON dict giong clone {speaker_emb, codes, style}; uu tien hon --voice")
    ap.add_argument("--style", default="tu_nhien")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-chars", type=int, default=256)
    ap.add_argument("--max-new-frames", type=int, default=300)
    args = ap.parse_args()

    text = DEFAULT_TEXT
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read().strip()
    n_words = len(text.split())

    result = {"ok": False, "n_words": n_words, "out": args.out}

    # --- device / torch ---
    t0 = time.time()
    import numpy as np
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    result["device"] = dev
    result["gpu_name"] = torch.cuda.get_device_name(0) if dev == "cuda" else None
    result["t_import_torch_s"] = round(time.time() - t0, 2)

    # --- load engine (tai model tu HuggingFace lan dau) ---
    _emit("PROGRESS:phase=load_model")
    t1 = time.time()
    from vieneu import Vieneu
    from vieneu_utils.phonemize_text import (
        normalize_to_chunks_v3_with_gaps, phonemize_text_with_emotions,
    )
    from vieneu_utils.core_utils import gaps_to_silence, join_audio_chunks
    tts = Vieneu(mode="v3turbo", device=dev)
    result["t_load_model_s"] = round(time.time() - t1, 2)
    result["backend"] = getattr(tts, "backend", "?")
    sr = getattr(tts, "sample_rate", 48000) or 48000
    result["sample_rate"] = sr

    # --- resolve giong (1 lan) ---
    # voice_file (clone dict) > voice (preset name) > default preset.
    if args.voice_file:
        with open(args.voice_file, encoding="utf-8") as f:
            rec = json.load(f)
        voice_arg = {
            "speaker_emb": np.asarray(rec["speaker_emb"], dtype=np.float32),
            "codes": None if rec.get("codes") is None else np.asarray(rec["codes"], dtype=np.int64),
            "style": rec.get("style", args.style),
        }
        result["voice_src"] = "clone_dict"
    elif args.voice:
        voice_arg = args.voice
        result["voice_src"] = "preset"
    else:
        voice_arg = None
        result["voice_src"] = "default"
    speaker_emb, ref_codes = tts._resolve_ref(
        voice=voice_arg, ref_audio=None, denoise=False, use_ref_codes=True)

    # --- infer THEO CHUNK (bao PROGRESS moi chunk) ---
    chunks, gaps = normalize_to_chunks_v3_with_gaps(text, max_chars=args.max_chars)
    n_chunks = len(chunks)
    result["n_chunks"] = n_chunks
    if n_chunks == 0:
        result["error"] = "text rong (0 chunk)"
        print("RESULT_JSON:" + json.dumps(result, ensure_ascii=False))
        return 1

    _emit(f"PROGRESS:phase=synth total={n_chunks}")
    t2 = time.time()
    all_wavs = []
    for i, chunk in enumerate(chunks):
        ph = phonemize_text_with_emotions(chunk)
        wav = tts.engine.infer(
            phonemes=ph, speaker_emb=speaker_emb, ref_codes=ref_codes,
            style=args.style, use_ref_codes=ref_codes is not None,
            temperature=args.temperature, max_new_frames=args.max_new_frames)
        if wav is not None and len(wav) > 0:
            all_wavs.append(wav)
        _emit(f"PROGRESS:chunk={i + 1} total={n_chunks}")
    result["t_infer_s"] = round(time.time() - t2, 2)

    if not all_wavs:
        result["error"] = "khong sinh duoc audio"
        print("RESULT_JSON:" + json.dumps(result, ensure_ascii=False))
        return 1

    audio = join_audio_chunks(all_wavs, sr, silence_ps=gaps_to_silence(gaps))

    # --- save ---
    t3 = time.time()
    tts.save(audio, args.out)
    result["t_save_s"] = round(time.time() - t3, 2)

    dur = len(audio) / sr
    result["audio_sec"] = round(dur, 2)
    result["rtf_x"] = round(dur / result["t_infer_s"], 2) if result["t_infer_s"] else None
    result["words_per_sec_synth"] = round(n_words / result["t_infer_s"], 2) if result["t_infer_s"] else None
    result["ok"] = True

    print("RESULT_JSON:" + json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
