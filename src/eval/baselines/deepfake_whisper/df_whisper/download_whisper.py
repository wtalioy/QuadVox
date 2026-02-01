"""Ensure Whisper tiny.en encoder checkpoint exists; download and extract if missing."""
from collections import OrderedDict
import os

from .commons import WHISPER_MODEL_WEIGHTS_PATH


def ensure_whisper_checkpoint() -> None:
    """Download Whisper tiny.en and save encoder to WHISPER_MODEL_WEIGHTS_PATH if missing."""
    if os.path.isfile(WHISPER_MODEL_WEIGHTS_PATH):
        return

    try:
        import whisper
        import torch
    except ImportError as e:
        raise ImportError(
            "Whisper checkpoint not found and openai-whisper is required to download it. "
            f"Install with: pip install openai-whisper. Expected path: {WHISPER_MODEL_WEIGHTS_PATH}"
        ) from e

    os.makedirs(os.path.dirname(WHISPER_MODEL_WEIGHTS_PATH), exist_ok=True)

    model = whisper.load_model("tiny.en")
    model_ckpt = OrderedDict()
    model_ckpt["model_state_dict"] = OrderedDict()
    for key, value in model.encoder.state_dict().items():
        model_ckpt["model_state_dict"][f"encoder.{key}"] = value
    model_ckpt["dims"] = model.dims
    torch.save(model_ckpt, WHISPER_MODEL_WEIGHTS_PATH)
