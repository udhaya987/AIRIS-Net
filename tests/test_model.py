import pytest
import torch
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airis.model import AIRISNet
from airis.losses import AIRISLoss


def test_model_initialization():
    """Verify model instantiates with correct parameter count and structure."""
    model_x1 = AIRISNet(in_channels=1, base_channels=48, scale_factor=1)
    params_x1 = sum(p.numel() for p in model_x1.parameters() if p.requires_grad)
    assert params_x1 > 200_000, f"Unexpected parameter count: {params_x1}"

    model_x2 = AIRISNet(in_channels=1, base_channels=48, scale_factor=2)
    params_x2 = sum(p.numel() for p in model_x2.parameters() if p.requires_grad)
    assert params_x2 > params_x1, "x2 model should have upsampling parameters"


def test_model_forward_dimensions_x1():
    """Verify forward pass output dimensions for scale=1 (denoising)."""
    model = AIRISNet(in_channels=1, base_channels=48, scale_factor=1)
    model.eval()
    x = torch.rand(2, 1, 128, 128)
    with torch.no_grad():
        out = model(x)

    assert "restored" in out
    assert "mask" in out
    assert "reliability" in out
    assert "routing_weights" in out

    assert out["restored"].shape == (2, 1, 128, 128)
    assert out["mask"].shape == (2, 1, 128, 128)
    assert out["reliability"].shape == (2, 1, 128, 128)
    assert out["routing_weights"].shape == (2, 3)


def test_model_forward_dimensions_x2():
    """Verify forward pass output dimensions for scale=2 (super-resolution)."""
    model = AIRISNet(in_channels=1, base_channels=48, scale_factor=2)
    model.eval()
    x = torch.rand(1, 1, 128, 128)
    with torch.no_grad():
        out = model(x)

    assert out["restored"].shape == (1, 1, 256, 256)
    assert out["mask"].shape == (1, 1, 256, 256)
    assert out["reliability"].shape == (1, 1, 256, 256)


def test_model_output_ranges():
    """Verify outputs are properly bounded in [0, 1]."""
    model = AIRISNet(in_channels=1, base_channels=48, scale_factor=1)
    model.eval()
    x = torch.rand(2, 1, 64, 64)
    with torch.no_grad():
        out = model(x)

    assert out["restored"].min() >= 0.0 and out["restored"].max() <= 1.0
    assert out["mask"].min() >= 0.0 and out["mask"].max() <= 1.0
    assert out["reliability"].min() >= 0.0 and out["reliability"].max() <= 1.0
    assert not torch.isnan(out["restored"]).any()
    assert not torch.isinf(out["restored"]).any()


def test_model_backward_pass():
    """Verify backward pass computes valid gradients without NaNs."""
    model = AIRISNet(in_channels=1, base_channels=48, scale_factor=1)
    criterion = AIRISLoss()
    x = torch.rand(2, 1, 64, 64)
    target = torch.rand(2, 1, 64, 64)

    out = model(x)
    loss, loss_dict = criterion(out, target, degraded_input=x)

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0.0

    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
