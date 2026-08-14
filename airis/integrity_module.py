import torch
import torch.nn as nn
from typing import Tuple


class IntegrityPreservingRestoration(nn.Module):
    """
    Integrity-Preserving Restoration Module for AIRIS-Net.
    Estimates a spatial restoration mask M and residual correction Delta I,
    selectively restoring degraded regions while preserving already reliable structures.
    Equation: I_restored = clamp(I_input + M * Delta I, 0.0, 1.0)
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 48, use_mask: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.use_mask = use_mask

        # Restoration mask head: predicts selective restoration confidence [0, 1]
        self.mask_head = nn.Sequential(
            nn.Conv2d(base_channels, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Residual / Reconstruction head: predicts reconstructed clean image [0, 1]
        self.residual_head = nn.Sequential(
            nn.Conv2d(base_channels, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(24, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, f_multi: torch.Tensor, img_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inputs:
            f_multi: (B, base_channels, H, W)
            img_input: (B, in_channels, H, W)
        Outputs:
            restored: (B, in_channels, H, W)
            mask: (B, 1, H, W)
            delta: (B, in_channels, H, W)
        """
        i_recon = self.residual_head(f_multi)
        # Residual correction delta = i_recon - img_input
        delta = i_recon - img_input

        if self.use_mask:
            mask = self.mask_head(f_multi)
            restored = torch.clamp(img_input + mask * delta, 0.0, 1.0)
        else:
            mask = torch.ones((img_input.size(0), 1, img_input.size(2), img_input.size(3)), device=img_input.device)
            restored = torch.clamp(img_input + delta, 0.0, 1.0)

        return restored, mask, delta
