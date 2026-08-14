import torch
import torch.nn as nn


class ShallowFeatureStem(nn.Module):
    """
    Shallow Feature Extraction module for AIRIS-Net.
    Extracts initial edge, texture, and structural feature representations (F0).
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 48):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, C, H, W)
        Output: F0 (B, base_channels, H, W)
        """
        return self.act(self.conv(x))
