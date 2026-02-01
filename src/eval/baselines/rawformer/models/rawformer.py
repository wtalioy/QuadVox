import torch
import torch.nn as nn
from .frontend import Frontend_SE
from .positional_aggregator import PositionalAggregator1D
from .classifier import RawformerClassifier


class Rawformer_SE(nn.Module):

    def __init__(self, device, transformer_hidden=660, sample_rate: int = 16000):
        super(Rawformer_SE, self).__init__()
        self.front_end = Frontend_SE(sinc_kernel_size=128, sample_rate=sample_rate)

        self.positional_embedding = PositionalAggregator1D(max_C = 64, max_ft=23*16, device=device)

        self.classifier = RawformerClassifier(C = 64, n_encoder = 2, transformer_hidden=transformer_hidden)

    def forward(self, x):
        x = self.front_end(x)
        x = self.positional_embedding(x)
        x = self.classifier(x)
        return x
