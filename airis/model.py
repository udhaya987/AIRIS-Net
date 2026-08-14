import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from airis.shallow_features import ShallowFeatureStem
from airis.degradation_encoder import DegradationSignatureEncoder
from airis.adaptive_router import AdaptiveRouter
from airis.local_expert import LocalCNNExpert
from airis.global_expert import GlobalContextExpert
from airis.frequency_expert import FrequencyExpert
from airis.fusion import DegradationConditionedFusion
from airis.multiscale import MultiScaleFeatureBlock
from airis.integrity_module import IntegrityPreservingRestoration
from airis.reliability import ReliabilityHead


class AIRISNet(nn.Module):
    """
    Adaptive Industrial Restoration & Integrity-Safeguarding Network (AIRIS-Net).
    
    Complete Architecture:
      Degraded Image
      -> Shallow Feature Extraction (F0)
      -> Degradation Signature Encoder (D)
      -> Adaptive Routing Controller (alpha_local, alpha_global, alpha_frequency)
      -> Local CNN Expert + Global Context Expert + Frequency Expert
      -> Degradation-Conditioned Feature Fusion
      -> Multi-Scale Feature Processing
      -> Integrity-Preserving Restoration Module (Mask M, Delta I)
      -> Reliability Map Estimation (R)
      -> Final Restored Output
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 48,
        degradation_dim: int = 64,
        use_local_expert: bool = True,
        use_global_expert: bool = True,
        use_frequency_expert: bool = True,
        use_adaptive_routing: bool = True,
        use_integrity_mask: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.degradation_dim = degradation_dim

        # Ablation switches
        self.use_local_expert = use_local_expert
        self.use_global_expert = use_global_expert
        self.use_frequency_expert = use_frequency_expert
        self.use_adaptive_routing = use_adaptive_routing
        self.use_integrity_mask = use_integrity_mask

        # 1. Shallow Feature Extraction
        self.stem = ShallowFeatureStem(in_channels=in_channels, base_channels=base_channels)

        # 2. Degradation Signature Encoder
        self.deg_encoder = DegradationSignatureEncoder(in_channels=in_channels, degradation_dim=degradation_dim)

        # 3. Adaptive Routing Controller
        self.router = AdaptiveRouter(degradation_dim=degradation_dim, num_experts=3)

        # 4. Specialized Experts
        if self.use_local_expert:
            self.local_expert = LocalCNNExpert(channels=base_channels, num_blocks=3)
        if self.use_global_expert:
            self.global_expert = GlobalContextExpert(channels=base_channels, window_size=8, num_heads=4, depth=2)
        if self.use_frequency_expert:
            self.frequency_expert = FrequencyExpert(channels=base_channels)

        # 5. Degradation-Conditioned Fusion
        self.fusion = DegradationConditionedFusion(channels=base_channels)

        # 6. Multi-Scale Feature Processing
        self.multiscale = MultiScaleFeatureBlock(channels=base_channels)

        # 7. Integrity-Preserving Restoration Module
        self.integrity_module = IntegrityPreservingRestoration(
            in_channels=in_channels,
            base_channels=base_channels,
            use_mask=use_integrity_mask
        )

        # 8. Reliability Map Head
        self.reliability_head = ReliabilityHead(base_channels=base_channels)

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass for AIRIS-Net.
        Input:
            x: (B, C, H, W) normalized to [0.0, 1.0]
        Output:
            dict containing:
                'restored': restored image tensor (B, C, H, W)
                'mask': restoration mask (B, 1, H, W)
                'reliability': reliability map (B, 1, H, W)
                'routing_weights': expert weights (B, 3)
                'degradation_embedding': latent vector D (B, degradation_dim)
                'diagnostic_scores': approximate [noise, blur, contrast] scores (B, 3)
        """
        B, C, H, W = x.shape

        # Step 1: Shallow Feature Extraction (F0)
        f0 = self.stem(x)

        # Step 2: Degradation Signature Encoding (D)
        d_emb, diag_scores = self.deg_encoder(x)

        # Step 3: Adaptive Routing
        if self.use_adaptive_routing:
            routing_weights = self.router(d_emb)
        else:
            # Uniform weights
            routing_weights = torch.ones((B, 3), device=x.device) / 3.0

        # Step 4: Expert Processing
        if self.use_local_expert:
            f_local = self.local_expert(f0)
        else:
            f_local = f0

        if self.use_global_expert:
            f_global = self.global_expert(f0)
        else:
            f_global = f0

        if self.use_frequency_expert:
            f_freq = self.frequency_expert(f0)
        else:
            f_freq = f0

        # Step 5: Degradation-Conditioned Fusion
        f_fused = self.fusion(f_local, f_global, f_freq, routing_weights, f0=f0)

        # Step 6: Multi-Scale Processing
        f_multi = self.multiscale(f_fused)

        # Step 7: Integrity-Preserving Restoration (Mask M and Delta I)
        restored, mask, delta = self.integrity_module(f_multi, x)

        # Step 8: Reliability Map Estimation (R)
        reliability = self.reliability_head(f_multi)

        return {
            "restored": restored,
            "mask": mask,
            "reliability": reliability,
            "routing_weights": routing_weights,
            "degradation_embedding": d_emb,
            "diagnostic_scores": diag_scores,
            "delta": delta
        }
