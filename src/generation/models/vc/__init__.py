import importlib
from .base import BaseVC

VC_MODEL_MAP = {
    "knnvc": "knnvc:KNNVC",
    "freevc": "freevc:FreeVC",
    "openvoice": "openvoice:OpenVoice",
}


def get_vc_model(name: str) -> type[BaseVC]:
    model_path = VC_MODEL_MAP[name]
    module_path, class_name = model_path.split(":", 1)
    module = importlib.import_module(f"{__name__}.{module_path}")
    return getattr(module, class_name)

def list_vc_models():
    return list(VC_MODEL_MAP.keys())

__all__ = ["BaseVC", "get_vc_model", "list_vc_models"]
