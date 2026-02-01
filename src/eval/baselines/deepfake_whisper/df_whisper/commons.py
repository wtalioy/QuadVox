"""Utility file for df_whisper toolkit."""
import os
import random

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_MODEL_WEIGHTS_PATH = os.path.join(
    _THIS_DIR, "models", "assets", "tiny_enc.en.pt"
)


def set_seed(seed: int):
    """Fix PRNG seed for reproducable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
