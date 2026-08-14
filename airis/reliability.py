import torch
import torch.nn as nn


class ReliabilityHead(nn.Module):
    """
    Predicted Reliability Head for AIRIS-Net.
    Outputs a per-pixel confidence / reliability map R in [0.0, 1.0].
    1 = High restoration reliability / fidelity
    0 = Low restoration reliability
    """
    def __init__(self, base_channels: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(base_channels, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, f_multi: torch.Tensor) -> torch.Tensor:
        """
        Input: f_multi (B, base_channels, H, W)
        Output: R (B, 1, H, W)
        """
        return self.net(f_multi)
