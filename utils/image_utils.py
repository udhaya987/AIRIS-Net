import os
from pathlib import Path
import numpy as np
import cv2
import torch
from typing import Union, Tuple, Optional


def load_image(path: Union[str, Path], grayscale: bool = True) -> np.ndarray:
    """
    Load image from file (.npy, .png, .jpg, .tiff, etc.) normalized to [0.0, 1.0], float32.
    Returns: (H, W) for grayscale or (H, W, C) for RGB.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {path}")

    if path.suffix.lower() == '.npy':
        img = np.load(path).astype(np.float32)
        if img.ndim == 3 and grayscale and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.ndim == 3 and not grayscale and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
    else:
        if grayscale:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image from {path}")
            img = img.astype(np.float32) / 255.0
        else:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not read image from {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Ensure range [0.0, 1.0]
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)
    return img


def save_image(img: Union[np.ndarray, torch.Tensor], save_path: Union[str, Path]) -> None:
    """
    Save image to file (.png, .jpg, .npy).
    Expects input in [0.0, 1.0].
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().squeeze().numpy()

    if save_path.suffix.lower() == '.npy':
        np.save(save_path, img.astype(np.float32))
        return

    img_uint8 = np.clip(img * 255.0, 0, 255).round().astype(np.uint8)
    if img_uint8.ndim == 3 and img_uint8.shape[-1] == 3:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(save_path), img_uint8)


def to_tensor(img: np.ndarray, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Convert (H, W) or (H, W, C) numpy array [0, 1] to (1, C, H, W) PyTorch tensor.
    """
    if img.ndim == 2:
        img = img[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
    elif img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))[np.newaxis, :, :, :]  # (1, C, H, W)
    return torch.from_numpy(img).float().to(device)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert (B, C, H, W) or (C, H, W) PyTorch tensor to (H, W) or (H, W, C) numpy array in [0, 1].
    """
    arr = tensor.detach().cpu().squeeze().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr, 0.0, 1.0)


def compute_change_map(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Compute absolute difference map between two images, normalized to [0, 1].
    """
    img1 = np.squeeze(img1).astype(np.float32)
    img2 = np.squeeze(img2).astype(np.float32)
    diff = np.abs(img1 - img2)
    return np.clip(diff, 0.0, 1.0)
