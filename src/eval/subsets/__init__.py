_MODEL_IMPORTS = {
    "BaseSubset": ".base",
    "PublicFigure": ".publicfigure",
    "News": ".news",
    "Podcast": ".podcast",
    "PartialFake": ".partialfake",
    "Audiobook": ".audiobook",
    "NoisySpeech": ".noisyspeech",
    "PhoneCall": ".phonecall",
    "Interview": ".interview",
    "PublicSpeech": ".publicspeech",
    "Movie": ".movie",
    "Emotional": ".emotional",
}

_MODEL_MAP = {
    "publicfigure": "PublicFigure",
    "news": "News",
    "podcast": "Podcast",
    "partialfake": "PartialFake",
    "audiobook": "Audiobook",
    "noisyspeech": "NoisySpeech",
    "phonecall": "PhoneCall",
    "interview": "Interview",
    "publicspeech": "PublicSpeech",
    "movie": "Movie",
    "emotional": "Emotional",
}

_cache = {}


def __getattr__(name):
    if name == "SUBSET_MAP":
        if "SUBSET_MAP" not in _cache:
            _cache["SUBSET_MAP"] = {
                key: getattr(__import__(_MODEL_IMPORTS[cls_name], fromlist=[cls_name], level=1), cls_name)
                for key, cls_name in _MODEL_MAP.items()
            }
        return _cache["SUBSET_MAP"]
    
    if name in _MODEL_IMPORTS:
        if name not in _cache:
            _cache[name] = getattr(__import__(_MODEL_IMPORTS[name], fromlist=[name], level=1), name)
        return _cache[name]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["BaseSubset", "SUBSET_MAP"]