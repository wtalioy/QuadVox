import importlib
from .base import BaseRawSubset

RAW_SUBSET_MAP = {
    "news": "news:News",
    "podcast": "podcast:Podcast",
    "movie": "movie:Movie",
    "phonecall": "phonecall:PhoneCall",
    "interview": "interview:Interview",
    "publicspeech": "publicspeech:PublicSpeech",
    "partialfake": "partialfake:PartialFake",
    "noisyspeech": "noisyspeech:NoisySpeech",
    "emotional": "emotional:Emotional",
}


def get_raw_subset(name: str) -> type[BaseRawSubset]:
    subset_path = RAW_SUBSET_MAP[name]
    module_path, class_name = subset_path.split(":", 1)
    module = importlib.import_module(f"{__name__}.{module_path}")
    return getattr(module, class_name)


def list_raw_subsets():
    return list(RAW_SUBSET_MAP.keys())

__all__ = [
    "BaseRawSubset",
    "RAW_SUBSET_MAP",
    "get_raw_subset",
    "list_raw_subsets",
]