def Vieneu(mode="v3turbo", **kwargs):
    """
    Factory function for VieNeu-TTS.

    Args:
        mode: 'v3turbo' — VieNeu-TTS v3 Turbo, 48 kHz. CPU runs torch-free via ONNX
              Runtime; GPU uses PyTorch. This build ships only the v3 Turbo engine
              (the legacy standard/fast/turbo/remote/xpu backends were removed to
              keep backend-model lean).
        **kwargs: Arguments for the chosen class.

    Returns:
        BaseVieneuTTS: A VieNeu-TTS v3 Turbo instance.
    """
    if mode != "v3turbo":
        raise ValueError(
            f"Unsupported mode '{mode}'. This backend-model build only supports "
            f"mode='v3turbo' (48 kHz; CPU=ONNX, GPU=PyTorch)."
        )
    from .v3turbo import V3TurboVieNeuTTS
    return V3TurboVieNeuTTS(**kwargs)
