import pytest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.degradation import RandomDegradationPipeline


def test_gaussian_noise_range():
    """Verify Gaussian noise produces valid array in [0, 1]."""
    pipe = RandomDegradationPipeline(seed=42)
    img = np.ones((64, 64), dtype=np.float32) * 0.5
    noisy, meta = pipe.apply_gaussian_noise(img, sigma=25.0)
    assert noisy.shape == img.shape
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0
    assert not np.array_equal(noisy, img)


def test_speckle_noise_multiplicative():
    """Verify speckle noise is multiplicative and zero areas remain zero."""
    pipe = RandomDegradationPipeline(seed=42)
    img = np.zeros((64, 64), dtype=np.float32)
    noisy, meta = pipe.apply_speckle_noise(img, variance=0.1)
    assert np.allclose(noisy, 0.0)

    img_active = np.ones((64, 64), dtype=np.float32) * 0.5
    noisy_active, _ = pipe.apply_speckle_noise(img_active, variance=0.1)
    assert not np.array_equal(noisy_active, img_active)
    assert noisy_active.min() >= 0.0 and noisy_active.max() <= 1.0


def test_resolution_degradation():
    """Verify spatial downsampling and dimension preservation options."""
    pipe = RandomDegradationPipeline(seed=42)
    img = np.random.rand(128, 128).astype(np.float32)
    deg_keep, _ = pipe.apply_resolution_degradation(img, scale_factor=2.0, keep_dim=True)
    assert deg_keep.shape == (128, 128)

    deg_low, _ = pipe.apply_resolution_degradation(img, scale_factor=2.0, keep_dim=False)
    assert deg_low.shape == (64, 64)


def test_combined_degradations():
    """Verify compound degradation pipelines."""
    pipe = RandomDegradationPipeline(seed=42)
    img = np.random.rand(128, 128).astype(np.float32)
    modes = ["gaussian_downsample", "speckle_downsample", "gaussian_speckle", "gaussian_speckle_downsample"]
    for m in modes:
        out, meta = pipe.apply_combined_degradation(img, mode=m, scale_factor=2.0)
        assert out.shape == (128, 128)
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_deterministic_seed():
    """Verify degradation repeatability when seed is fixed."""
    img = np.ones((64, 64), dtype=np.float32) * 0.5
    pipe = RandomDegradationPipeline()

    pipe.set_seed(123)
    out1, _ = pipe(img, "gaussian", sigma=25.0)

    pipe.set_seed(123)
    out2, _ = pipe(img, "gaussian", sigma=25.0)

    assert np.allclose(out1, out2)
