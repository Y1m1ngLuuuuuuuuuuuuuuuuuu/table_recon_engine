import torch
from torch import nn


class StripPooling(nn.Module):
    """Pure tensor strip pooling attention for table row/column cues.

    Given X in shape (B, C, H, W), it pools along width and height separately,
    broadcasts the two strip responses back to the full feature map, then uses a
    sigmoid attention map with a residual connection.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"StripPooling expects a 4D tensor, got {x.shape}")

        horizontal = torch.mean(x, dim=3, keepdim=True)  # (B, C, H, 1)
        vertical = torch.mean(x, dim=2, keepdim=True)  # (B, C, 1, W)

        horizontal = horizontal.expand_as(x)
        vertical = vertical.expand_as(x)

        attention = torch.sigmoid(horizontal + vertical)
        return x * attention + x
