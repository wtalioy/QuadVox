import torch
import torch.nn as nn


class PositionalAggregator1D(nn.Module):

    def __init__(self, max_C: int, max_ft: int, device):
        """_summary_

        Args:
            max_channels (int): for HFM, max size of C
            max_ft (int): for HFM, max size of f*t
        """
        super(PositionalAggregator1D, self).__init__()

        self.flattener = nn.Flatten(start_dim=-2, end_dim=-1)

        # ------------------ positional encoding -------------------- #
        x = torch.arange(1, max_ft - 1, device=device).float()
        x = x.float().unsqueeze(1)
        _2i = torch.arange(0, max_C, step=2, device=device).float().unsqueeze(0)

        self.encoding = torch.zeros(max_ft, max_C, device=device, requires_grad=False)
        self.encoding[1:-1, 0::2] = torch.sin(x / (10000 ** (_2i / max_C)))
        self.encoding[1:-1, 1::2] = torch.cos(x / (10000 ** (_2i / max_C)))

    def forward(self, HFM):
        batch, C, f, t = HFM.shape
        out = self.flattener(HFM).transpose(1, 2)  # [batch, f*t, C]
        out = out + self.encoding[: f * t, :C].unsqueeze(0)
        return out
