import importlib
from .base import BaseTTS

TTS_MODEL_MAP = {
    "vits": "vits:VITS",
    "xttsv2": "xttsv2:XTTSv2",
    "yourtts": "yourtts:YourTTS",
    "tacotron2": "tacotron2:Tacotron2",
    "bark": "bark:Bark",
    "melotts": "melotts:MeloTTS",
    "elevenlabs": "elevenlabs_tts:ElevenLabsTTS",
    "geminitts": "gemini_tts:GeminiTTS",
    "gpt4omini": "gpt4omini_tts:GPT4oMiniTTS",
    "indextts": "indextts:IndexTTS",
    "cosyvoice": "cosyvoice:CosyVoice",
    "f5tts": "f5tts:F5TTS",
}


def get_tts_model(name: str) -> type[BaseTTS]:
    model_path = TTS_MODEL_MAP[name]
    module_path, class_name = model_path.split(":", 1)
    module = importlib.import_module(f"{__name__}.{module_path}")
    return getattr(module, class_name)

def list_tts_models():
    return list(TTS_MODEL_MAP.keys())

__all__ = ["BaseTTS", "TTS_MODEL_MAP", "get_tts_model", "list_tts_models"]
