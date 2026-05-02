import torch
from torch import nn

from table_recon_engine.custom_ops import StripPooling


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        return self.relu(out)


class VisionEncoder(nn.Module):
    """Lightweight ResNet-style CNN with a custom StripPooling module."""

    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(32, 32, blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=2)
        self.strip_pooling = StripPooling()
        self.proj = nn.Conv2d(128, out_dim, kernel_size=1)
        self.out_dim = out_dim

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.stem(images)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.strip_pooling(features)
        features = self.proj(features)

        bsz, channels, height, width = features.shape
        memory = features.flatten(2).permute(2, 0, 1).contiguous()
        memory_mask = images.new_zeros((bsz, height * width), dtype=torch.bool)
        return memory, memory_mask
