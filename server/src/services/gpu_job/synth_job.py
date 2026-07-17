"""synth_job.py — chạy TRÊN máy GPU Vast.ai (POC).

Nhận 1 đoạn text (mặc định ~360 từ), nạp engine VieNeu v3 Turbo (PyTorch/CUDA),
sinh audio, lưu WAV, và in TIMING từng bước ra stdout dạng JSON dòng cuối
(prefix "RESULT_JSON:") để orchestrator ở máy local đọc lại.

Chạy:  python synth_job.py --out /workspace/out.wav [--text-file f.txt] [--voice "Ngọc Lan"]

Chỉ phụ thuộc engine của tác giả (package `vieneu` + `vieneu_utils`) — KHÔNG import
backend `src` đang refactor. Watermark tắt mặc định (khỏi cài `perth`).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/out.wav")
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--voice", default=None, help="ten voice preset; bo trong = default")
    ap.add_argument("--style", default="tu_nhien")
    args = ap.parse_args()

    text = DEFAULT_TEXT
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read().strip()
    n_words = len(text.split())

    result = {"ok": False, "n_words": n_words, "out": args.out}

    # --- device / torch ---
    t0 = time.time()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    result["device"] = dev
    result["gpu_name"] = torch.cuda.get_device_name(0) if dev == "cuda" else None
    result["t_import_torch_s"] = round(time.time() - t0, 2)

    # --- load engine (tai model tu HuggingFace lan dau) ---
    t1 = time.time()
    from vieneu import Vieneu
    tts = Vieneu(mode="v3turbo", device=dev)
    result["t_load_model_s"] = round(time.time() - t1, 2)
    result["backend"] = getattr(tts, "backend", "?")
    result["sample_rate"] = getattr(tts, "sample_rate", None)

    # --- infer ---
    t2 = time.time()
    kw = {"style": args.style, "apply_watermark": False}
    if args.voice:
        kw["voice"] = args.voice
    audio = tts.infer(text, **kw)
    result["t_infer_s"] = round(time.time() - t2, 2)

    # --- save ---
    t3 = time.time()
    tts.save(audio, args.out)
    result["t_save_s"] = round(time.time() - t3, 2)

    sr = result["sample_rate"] or 48000
    dur = len(audio) / sr
    result["audio_sec"] = round(dur, 2)
    result["rtf_x"] = round(dur / result["t_infer_s"], 2) if result["t_infer_s"] else None
    result["words_per_sec_synth"] = round(n_words / result["t_infer_s"], 2) if result["t_infer_s"] else None
    result["ok"] = True

    print("RESULT_JSON:" + json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
