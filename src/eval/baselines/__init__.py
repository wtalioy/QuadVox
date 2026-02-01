import importlib
from .base import Baseline

BASELINE_MODEL_MAP = {
    "aasist": "aasist.aasist:AASIST",
    "aasist-l": "aasist.aasist:AASIST_L",
    "rapt": "rapt.rapt:RAPT",
    "res-tssdnet": "TSSDNet.tssdnet:Res_TSSDNet",
    "inc-tssdnet": "TSSDNet.tssdnet:Inc_TSSDNet",
    "rawnet2": "RawNet2.rawnet2:RawNet2",
    "rawgat-st": "RawGAT_ST.rawgat_st:RawGAT_ST",
    "rawformer": "rawformer.rawformer:Rawformer",
    "whisper-specrnet": "deepfake_whisper.whisper_features:WhisperSpecRNet",
}


def get_baseline_model(name: str) -> type[Baseline]:
    model_path = BASELINE_MODEL_MAP[name]
    module_path, class_name = model_path.split(":", 1)
    module = importlib.import_module(f"{__name__}.{module_path}")
    return getattr(module, class_name)

def list_baseline_models():
    return list(BASELINE_MODEL_MAP.keys())

__all__ = ["Baseline", "BASELINE_MODEL_MAP", "get_baseline_model", "list_baseline_models"]