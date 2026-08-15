import pytest
import numpy as np
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics import calculate_psnr, calculate_ssim, ssim_torch, calculate_lpips


def test_psnr_identity():
    """Verify PSNR of identical images is infinite or capped."""
    img = np.random.rand(64, 64).astype(np.float32)
    psnr_val = calculate_psnr(img, img)
    assert psnr_val == float("inf") or psnr_val > 80.0


def test_psnr_known_noise():
    """Verify PSNR decreases as noise increases."""
    clean = np.ones((64, 64), dtype=np.float32) * 0.5
    noisy1 = np.clip(clean + np.random.normal(0, 0.05, clean.shape), 0, 1).astype(np.float32)
    noisy2 = np.clip(clean + np.random.normal(0, 0.20, clean.shape), 0, 1).astype(np.float32)

    psnr1 = calculate_psnr(noisy1, clean)
    psnr2 = calculate_psnr(noisy2, clean)
    assert psnr1 > psnr2, "Lower noise should yield higher PSNR"


def test_ssim_range():
    """Verify SSIM output is in [-1, 1] and 1.0 for identical images."""
    clean = np.random.rand(64, 64).astype(np.float32)
    ssim_same = calculate_ssim(clean, clean)
    assert np.isclose(ssim_same, 1.0, atol=1e-4)

    noisy = np.random.rand(64, 64).astype(np.float32)
    ssim_noisy = calculate_ssim(noisy, clean)
    assert -1.0 <= ssim_noisy <= 1.0


def test_ssim_torch():
    """Verify differentiable PyTorch SSIM loss computation."""
    t1 = torch.rand(2, 1, 64, 64)
    t2 = torch.rand(2, 1, 64, 64)
    val = ssim_torch(t1, t2)
    assert val.shape == ()
    assert 0.0 <= val.item() <= 1.0


def test_lpips():
    """Verify LPIPS computation returns valid float score."""
    img1 = np.random.rand(64, 64).astype(np.float32)
    img2 = np.random.rand(64, 64).astype(np.float32)
    lpips_val = calculate_lpips(img1, img2, device=torch.device("cpu"))
    if lpips_val is not None:
        assert isinstance(lpips_val, float)
        assert lpips_val >= 0.0
