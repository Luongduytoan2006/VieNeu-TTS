"""Vendored VieNeu-TTS v3 Turbo PyTorch engine (model + inference).
"""
from .configuration_v3_turbo import VieNeuV3TurboConfig
from .modeling_v3_turbo import VieNeuV3TurboForTTS
from .inference_v3_turbo import VieNeuTTSv3Turbo

__all__ = ["VieNeuV3TurboConfig", "VieNeuV3TurboForTTS", "VieNeuTTSv3Turbo"]
