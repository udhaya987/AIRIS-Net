import torch
import torch.nn as nn


class MultiScaleFeatureBlock(nn.Module):
    """
    Multi-Scale Feature Processing Block for AIRIS-Net.
    Captures multi-receptive field structural details using parallel dilated convolutions.
    """
    def __init__(self, channels: int = 48):
        super().__init__()
        self.branch1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1, bias=False)
        self.branch2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.branch3 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, bias=False)

        self.act = nn.GELU()
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)
        self.out_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: F_fused (B, channels, H, W)
        Output: F_multi (B, channels, H, W)
        """
        res = x
        b1 = self.act(self.branch1(x))
        b2 = self.act(self.branch2(x))
        b3 = self.act(self.branch3(x))

        concat = torch.cat([b1, b2, b3], dim=1)
        out = self.out_act(self.fuse(concat))

        return res + out
