import torch
import torch.nn as nn
from typing import Tuple, Optional


class DegradationSignatureEncoder(nn.Module):
    """
    Degradation Signature Encoder for AIRIS-Net.
    Learns a compact degradation latent embedding (D) without requiring explicit degradation labels.
    Also provides an optional diagnostic head predicting approximate noise, blur, and contrast scores.
    """
    def __init__(self, in_channels: int = 1, degradation_dim: int = 64):
        super().__init__()
        self.degradation_dim = degradation_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),  # /2
            nn.GELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),  # /4
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, degradation_dim),
            nn.GELU()
        )

        # Optional diagnostic head for noise, blur, contrast score visualization
        self.diagnostic_head = nn.Sequential(
            nn.Linear(degradation_dim, 32),
            nn.GELU(),
            nn.Linear(32, 3),
            nn.Sigmoid()  # normalized scores [0, 1]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: x (B, C, H, W)
        Output:
            d_emb: (B, degradation_dim)
            diag_scores: (B, 3) [noise_score, blur_score, contrast_score]
        """
        d_emb = self.encoder(x)
        diag_scores = self.diagnostic_head(d_emb)
        return d_emb, diag_scores
