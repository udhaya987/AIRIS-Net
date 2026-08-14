import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from utils.edge_utils import compute_sobel_edges_torch
from utils.frequency_utils import compute_fft_magnitude_torch
from utils.metrics import ssim_torch


class CharbonnierLoss(nn.Module):
    """
    Differentiable Charbonnier loss (smooth L1 / robust penalty).
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps2)
        return torch.mean(loss)


class AIRISLoss(nn.Module):
    """
    Multi-Objective Loss for AIRIS-Net:
      1. Charbonnier Reconstruction Loss (L_char)
      2. Sobel Edge Gradient Loss (L_edge)
      3. Structural Similarity Loss (L_ssim)
      4. Frequency Log-Magnitude Loss (L_freq)
      5. Identity / Structure Preservation Loss (L_identity)
      6. Restoration Mask Supervision Loss (L_mask)
      7. Reliability Map Supervision Loss (L_reliability)
    """
    def __init__(
        self,
        w_char: float = 1.0,
        w_edge: float = 0.1,
        w_ssim: float = 0.1,
        w_freq: float = 0.05,
        w_identity: float = 0.1,
        w_mask: float = 0.05,
        w_reliability: float = 0.05,
        k_reliability: float = 10.0
    ):
        super().__init__()
        self.w_char = w_char
        self.w_edge = w_edge
        self.w_ssim = w_ssim
        self.w_freq = w_freq
        self.w_identity = w_identity
        self.w_mask = w_mask
        self.w_reliability = w_reliability
        self.k_reliability = k_reliability

        self.char_loss = CharbonnierLoss(eps=1e-3)
        self.l1_loss = nn.L1Loss()

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        clean_target: torch.Tensor,
        degraded_input: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Inputs:
            model_outputs: dict from AIRISNet forward pass
            clean_target: (B, C, H, W)
            degraded_input: (B, C, H, W)
        Outputs:
            total_loss: scalar tensor
            loss_dict: dict of individual loss values (floats)
        """
        restored = model_outputs["restored"]
        pred_mask = model_outputs["mask"]
        pred_rel = model_outputs["reliability"]

        # 1. Charbonnier Reconstruction Loss
        l_char = self.char_loss(restored, clean_target)

        # 2. Edge Loss using Sobel filters
        edge_pred = compute_sobel_edges_torch(restored)
        edge_target = compute_sobel_edges_torch(clean_target)
        l_edge = self.l1_loss(edge_pred, edge_target)

        # 3. SSIM Loss
        ssim_val = ssim_torch(restored, clean_target)
        l_ssim = 1.0 - ssim_val

        # 4. Frequency Loss (log magnitude spectrum)
        fft_pred = compute_fft_magnitude_torch(restored)
        fft_target = compute_fft_magnitude_torch(clean_target)
        l_freq = self.l1_loss(fft_pred, fft_target)

        # 5. Identity / Preservation Loss
        # Region where degraded is already close to clean (< 0.03 diff) should be preserved
        abs_diff_deg = torch.abs(degraded_input - clean_target)
        m_gt = torch.clamp(abs_diff_deg * 10.0, 0.0, 1.0)
        preservation_mask = torch.clamp(1.0 - m_gt, 0.0, 1.0)
        l_identity = torch.mean(preservation_mask * torch.abs(restored - degraded_input))

        # 6. Restoration Mask Supervision
        m_target = torch.clamp(torch.mean(abs_diff_deg, dim=1, keepdim=True) * 5.0, 0.0, 1.0)
        l_mask = self.l1_loss(pred_mask, m_target)

        # 7. Reliability Map Supervision
        # Restoration error with detached restored prediction
        restoration_err = torch.mean(torch.abs(restored.detach() - clean_target), dim=1, keepdim=True)
        r_target = torch.clamp(torch.exp(-self.k_reliability * restoration_err), 0.0, 1.0)
        l_reliability = self.l1_loss(pred_rel, r_target)

        # Total Weighted Loss
        total_loss = (
            self.w_char * l_char
            + self.w_edge * l_edge
            + self.w_ssim * l_ssim
            + self.w_freq * l_freq
            + self.w_identity * l_identity
            + self.w_mask * l_mask
            + self.w_reliability * l_reliability
        )

        loss_dict = {
            "total": float(total_loss.item()),
            "char": float(l_char.item()),
            "edge": float(l_edge.item()),
            "ssim": float(l_ssim.item()),
            "freq": float(l_freq.item()),
            "identity": float(l_identity.item()),
            "mask": float(l_mask.item()),
            "reliability": float(l_reliability.item())
        }

        return total_loss, loss_dict
