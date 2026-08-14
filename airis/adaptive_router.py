import torch
import torch.nn as nn


class AdaptiveRouter(nn.Module):
    """
    Adaptive Routing Controller for AIRIS-Net.
    Maps degradation signature D to expert routing weights:
    (alpha_local, alpha_global, alpha_frequency) such that their sum equals 1.0.
    """
    def __init__(self, degradation_dim: int = 64, num_experts: int = 3):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Sequential(
            nn.Linear(degradation_dim, 32),
            nn.GELU(),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, d_emb: torch.Tensor) -> torch.Tensor:
        """
        Input: d_emb (B, degradation_dim)
        Output: weights (B, num_experts) -> [alpha_local, alpha_global, alpha_frequency]
        """
        return self.router(d_emb)
