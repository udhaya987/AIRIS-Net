import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class IntegrityPreservingRestoration(nn.Module):
    """
    Integrity-Preserving Restoration Module for AIRIS-Net.
    Estimates a spatial restoration mask M and residual correction Delta I,
    selectively restoring degraded regions while preserving already reliable structures.
    Supports both same-resolution (scale=1) and super-resolution (scale >= 2).
    Equation: I_restored = clamp(I_input_scaled + M * Delta I, 0.0, 1.0)
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 48,
        use_mask: bool = True,
        scale_factor: int = 1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.use_mask = use_mask
        self.scale_factor = scale_factor

        # Feature upsampling for super-resolution (x2, etc.)
        if scale_factor > 1:
            self.upsample = nn.Sequential(
                nn.Conv2d(base_channels, base_channels * (scale_factor ** 2), kernel_size=3, padding=1),
                nn.PixelShuffle(scale_factor),
                nn.GELU()
            )
        else:
            self.upsample = nn.Identity()

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
            restored: (B, in_channels, H * scale, W * scale)
            mask: (B, 1, H * scale, W * scale)
            delta: (B, in_channels, H * scale, W * scale)
        """
        f_feat = self.upsample(f_multi)
        i_recon = self.residual_head(f_feat)

        if self.scale_factor > 1:
            img_input_scaled = F.interpolate(
                img_input,
                size=(f_feat.shape[2], f_feat.shape[3]),
                mode='bicubic',
                align_corners=False
            )
        else:
            img_input_scaled = img_input

        # Residual correction delta = i_recon - img_input_scaled
        delta = i_recon - img_input_scaled

        if self.use_mask:
            mask = self.mask_head(f_feat)
            restored = torch.clamp(img_input_scaled + mask * delta, 0.0, 1.0)
        else:
            mask = torch.ones(
                (img_input_scaled.size(0), 1, img_input_scaled.size(2), img_input_scaled.size(3)),
                device=img_input.device
            )
            restored = torch.clamp(img_input_scaled + delta, 0.0, 1.0)

        return restored, mask, delta
