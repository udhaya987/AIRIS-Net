import torch
import torch.nn as nn


class LocalResidualBlock(nn.Module):
    """
    Lightweight residual CNN block with standard Conv + Depthwise-Separable Conv.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.act1 = nn.GELU()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pwconv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act1(self.conv1(x))
        out = self.dwconv(out)
        out = self.act2(self.pwconv(out))
        return res + out


class LocalCNNExpert(nn.Module):
    """
    Local CNN Expert for AIRIS-Net.
    Specialized in restoring sharp edges, high-frequency textures, and localized noise.
    """
    def __init__(self, channels: int = 48, num_blocks: int = 3):
        super().__init__()
        self.blocks = nn.Sequential(*[LocalResidualBlock(channels) for _ in range(num_blocks)])
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: F0 (B, channels, H, W)
        Output: F_local (B, channels, H, W)
        """
        return x + self.out_conv(self.blocks(x))
