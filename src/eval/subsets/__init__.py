import importlib
from .base import BaseSubset

SUBSET_MODEL_MAP = {
    "publicfigure": "publicfigure:PublicFigure",
    "news": "news:News",
    "podcast": "podcast:Podcast",
    "partialfake": "partialfake:PartialFake",
    "audiobook": "audiobook:Audiobook",
    "noisyspeech": "noisyspeech:NoisySpeech",
    "phonecall": "phonecall:PhoneCall",
    "interview": "interview:Interview",
    "publicspeech": "publicspeech:PublicSpeech",
    "movie": "movie:Movie",
    "emotional": "emotional:Emotional",
}


def get_subset_model(name: str) -> type[BaseSubset]:
    model_path = SUBSET_MODEL_MAP[name]
    module_path, class_name = model_path.split(":", 1)
    module = importlib.import_module(f"{__name__}.{module_path}")
    return getattr(module, class_name)

def list_subset_models():
    return list(SUBSET_MODEL_MAP.keys())

__all__ = ["BaseSubset", "SUBSET_MODEL_MAP", "get_subset_model", "list_subset_models"]