import torch
import torch.nn as nn
from typing import Optional


class DegradationConditionedFusion(nn.Module):
    """
    Degradation-Conditioned Feature Fusion module for AIRIS-Net.
    Dynamically weights and merges expert representations using routing coefficients,
    followed by multi-layer non-linear refinement with residual connection from F0.
    """
    def __init__(self, channels: int = 48):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GELU()
        )

    def forward(
        self,
        f_local: torch.Tensor,
        f_global: torch.Tensor,
        f_freq: torch.Tensor,
        routing_weights: torch.Tensor,
        f0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Inputs:
            f_local: (B, C, H, W)
            f_global: (B, C, H, W)
            f_freq: (B, C, H, W)
            routing_weights: (B, 3) -> [alpha_local, alpha_global, alpha_freq]
            f0: (B, C, H, W) optional shallow features for skip connection
        Output:
            f_fused: (B, C, H, W)
        """
        # Broadcast weights to (B, 1, 1, 1)
        w_local = routing_weights[:, 0].view(-1, 1, 1, 1)
        w_global = routing_weights[:, 1].view(-1, 1, 1, 1)
        w_freq = routing_weights[:, 2].view(-1, 1, 1, 1)

        # Weighted combination
        f_weighted = w_local * f_local + w_global * f_global + w_freq * f_freq

        # Refinement
        f_refined = self.refine(f_weighted)

        # Residual connection from F0 if provided
        if f0 is not None:
            f_refined = f_refined + f0

        return f_refined
