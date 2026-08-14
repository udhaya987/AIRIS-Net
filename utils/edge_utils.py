import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Union, Tuple


def compute_sobel_edges_np(img: np.ndarray) -> np.ndarray:
    """
    Compute Sobel edge gradient magnitude for a 2D numpy array [0, 1].
    Returns edge magnitude in [0, 1].
    """
    img = np.squeeze(img).astype(np.float32)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    
    max_val = magnitude.max()
    if max_val > 1e-6:
        magnitude = magnitude / max_val
    return np.clip(magnitude, 0.0, 1.0)


def compute_sobel_edges_torch(tensor: torch.Tensor) -> torch.Tensor:
    """
    Differentiable Sobel edge magnitude computation for PyTorch tensor (B, C, H, W).
    """
    device = tensor.device
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]], device=device).view(1, 1, 3, 3)
    
    b, c, h, w = tensor.shape
    if c > 1:
        # Convert RGB to grayscale weights
        tensor_gray = 0.2989 * tensor[:, 0:1, :, :] + 0.5870 * tensor[:, 1:2, :, :] + 0.1140 * tensor[:, 2:3, :, :]
    else:
        tensor_gray = tensor

    # Pad tensor to preserve spatial dimensions
    padded = F.pad(tensor_gray, (1, 1, 1, 1), mode='reflect')
    gx = F.conv2d(padded, sobel_x)
    gy = F.conv2d(padded, sobel_y)
    
    magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    return magnitude


def edge_consistency_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate edge consistency metric between two images based on normalized cross-correlation / L1 of edge maps.
    Returns score between 0.0 (inconsistent) and 1.0 (identical edge structure).
    """
    e1 = compute_sobel_edges_np(img1)
    e2 = compute_sobel_edges_np(img2)
    
    l1_diff = np.mean(np.abs(e1 - e2))
    score = np.exp(-5.0 * l1_diff)
    return float(np.clip(score, 0.0, 1.0))
