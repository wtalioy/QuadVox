_MODEL_IMPORTS = {
    "Baseline": ".base",
    "AASIST": ".aasist.aasist",
    "AASIST_L": ".aasist.aasist",
    "RAPT": ".rapt.rapt",
    "Res_TSSDNet": ".TSSDNet.tssdnet",
    "Inc_TSSDNet": ".TSSDNet.tssdnet",
    "RawNet2": ".RawNet2.rawnet2",
    "RawGAT_ST": ".RawGAT_ST.rawgat_st",
}

_MODEL_MAP = {
    "aasist": "AASIST",
    "aasist-l": "AASIST_L",
    "rapt": "RAPT",
    "res-tssdnet": "Res_TSSDNet",
    "inc-tssdnet": "Inc_TSSDNet",
    "rawnet2": "RawNet2",
    "rawgat-st": "RawGAT_ST",
}

_cache = {}


def __getattr__(name):
    if name == "BASELINE_MAP":
        if "BASELINE_MAP" not in _cache:
            _cache["BASELINE_MAP"] = {
                key: getattr(__import__(_MODEL_IMPORTS[cls_name], fromlist=[cls_name], level=1), cls_name)
                for key, cls_name in _MODEL_MAP.items()
            }
        return _cache["BASELINE_MAP"]
    
    if name in _MODEL_IMPORTS:
        if name not in _cache:
            _cache[name] = getattr(__import__(_MODEL_IMPORTS[name], fromlist=[name], level=1), name)
        return _cache[name]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Baseline", "BASELINE_MAP"]