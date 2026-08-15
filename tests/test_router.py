import pytest
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airis.adaptive_router import AdaptiveRouter
from airis.degradation_encoder import DegradationSignatureEncoder


def test_router_weight_normalization():
    """Verify router weights sum to 1.0 via Softmax."""
    router = AdaptiveRouter(degradation_dim=64, num_experts=3)
    d_emb = torch.randn(8, 64)
    weights = router(d_emb)

    assert weights.shape == (8, 3)
    assert (weights >= 0.0).all(), "Negative weights found"
    assert torch.allclose(weights.sum(dim=-1), torch.ones(8), atol=1e-5)


def test_degradation_encoder_and_router_integration():
    """Verify degradation encoder feeds into router cleanly."""
    encoder = DegradationSignatureEncoder(in_channels=1, degradation_dim=64)
    router = AdaptiveRouter(degradation_dim=64, num_experts=3)

    x = torch.rand(4, 1, 128, 128)
    d_emb, diag_scores = encoder(x)

    assert d_emb.shape == (4, 64)
    assert diag_scores.shape == (4, 3)
    assert (diag_scores >= 0.0).all() and (diag_scores <= 1.0).all()

    weights = router(d_emb)
    assert weights.shape == (4, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-5)
