import torch
import numpy as np
from typing import Union, Tuple, Dict


def compute_fft_magnitude_np(img: np.ndarray) -> np.ndarray:
    """
    Compute log-magnitude 2D FFT spectrum for visualization.
    Returns normalized spectrum in [0, 1].
    """
    img = np.squeeze(img).astype(np.float32)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
    
    # Normalize to [0, 1]
    norm_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (
        magnitude_spectrum.max() - magnitude_spectrum.min() + 1e-8
    )
    return np.clip(norm_spectrum, 0.0, 1.0)


def compute_fft_magnitude_torch(tensor: torch.Tensor) -> torch.Tensor:
    """
    Differentiable 2D FFT log magnitude for PyTorch tensor (B, C, H, W).
    """
    fft = torch.fft.fft2(tensor)
    fft_shifted = torch.fft.fftshift(fft)
    magnitude = torch.abs(fft_shifted)
    log_magnitude = torch.log(magnitude + 1e-6)
    return log_magnitude


def frequency_consistency_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate frequency consistency metric between two images based on log FFT spectrum similarity.
    Returns score in [0.0, 1.0].
    """
    spec1 = compute_fft_magnitude_np(img1)
    spec2 = compute_fft_magnitude_np(img2)
    
    diff = np.mean(np.abs(spec1 - spec2))
    score = np.exp(-3.0 * diff)
    return float(np.clip(score, 0.0, 1.0))
