import torch
from torch import nn


class RegressionHead(nn.Module):
    """Predicts normalized cell bounding boxes from decoder hidden states."""

    def __init__(self, d_model: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

    def forward(self, decoder_states: torch.Tensor) -> torch.Tensor:
        raw = self.layers(decoder_states)
        x0y0 = torch.min(raw[..., :2], raw[..., 2:])
        x1y1 = torch.max(raw[..., :2], raw[..., 2:])
        return torch.cat([x0y0, x1y1], dim=-1)
