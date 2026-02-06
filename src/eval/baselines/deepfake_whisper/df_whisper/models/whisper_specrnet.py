import torch

from .. import frontends
from ..commons import WHISPER_MODEL_WEIGHTS_PATH
from ..download_whisper import ensure_whisper_checkpoint
from .whisper_main import ModelDimensions, Whisper, log_mel_spectrogram
from .specrnet import SpecRNet


class WhisperSpecRNet(SpecRNet):
    def __init__(self, input_channels, freeze_encoder, **kwargs):
        super().__init__(input_channels=input_channels, **kwargs)

        self.device = kwargs["device"]
        ensure_whisper_checkpoint()
        try:
            checkpoint = torch.load(WHISPER_MODEL_WEIGHTS_PATH, weights_only=False)
        except TypeError:
            checkpoint = torch.load(WHISPER_MODEL_WEIGHTS_PATH)
        dims = ModelDimensions(**checkpoint["dims"].__dict__)
        model = Whisper(dims)
        model = model.to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.whisper_model = model
        if freeze_encoder:
            for param in self.whisper_model.parameters():
                param.requires_grad = False
        self.to(self.device)

    def compute_whisper_features(self, x):
        specs = []
        for sample in x:
            specs.append(log_mel_spectrogram(sample))
        x = torch.stack(specs)
        x = self.whisper_model(x)

        x = x.permute(0, 2, 1)  # (bs, frames, 3 x n_lfcc)
        x = x.unsqueeze(1)  # (bs, 1, frames, 3 x n_lfcc)
        x = x.repeat(
            (1, 1, 1, 2)
        )  # (bs, 1, frames, 3 x n_lfcc) -> (bs, 1, frames, 3000)
        return x

    def forward(self, x):
        x = self.compute_whisper_features(x)
        out = self._compute_embedding(x)
        return out


class WhisperMultiFrontSpecRNet(WhisperSpecRNet):
    def __init__(self, input_channels, freeze_encoder, **kwargs):
        super().__init__(
            input_channels=input_channels,
            freeze_encoder=freeze_encoder,
            **kwargs,
        )
        self.frontend = frontends.get_frontend(kwargs["frontend_algorithm"])
        self.to(self.device)

    def forward(self, x):
        frontend_x = self.frontend(x)
        x = self.compute_whisper_features(x)

        x = torch.cat([x, frontend_x], 1)
        out = self._compute_embedding(x)
        return out
