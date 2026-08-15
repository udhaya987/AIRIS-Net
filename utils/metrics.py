import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Union, Tuple, Optional

# Lazy LPIPS model cache
_LPIPS_MODEL = None
_LPIPS_AVAILABLE = True


def calculate_psnr(
    img1: Union[np.ndarray, torch.Tensor],
    img2: Union[np.ndarray, torch.Tensor],
    data_range: float = 1.0
) -> float:
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.
    Supports numpy arrays or PyTorch tensors normalized to [0, 1] or [0, 255].
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    img1 = np.squeeze(img1).astype(np.float32)
    img2 = np.squeeze(img2).astype(np.float32)

    # Ensure identical shape (crop to common min shape if slight difference)
    if img1.shape != img2.shape:
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]

    max_val = max(img1.max(), img2.max())
    if max_val > 1.5:
        data_range = 255.0

    return float(peak_signal_noise_ratio(img2, img1, data_range=data_range))


def calculate_ssim(
    img1: Union[np.ndarray, torch.Tensor],
    img2: Union[np.ndarray, torch.Tensor],
    data_range: float = 1.0
) -> float:
    """
    Calculate SSIM (Structural Similarity Index) between two images.
    Supports numpy arrays or PyTorch tensors.
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    img1 = np.squeeze(img1).astype(np.float32)
    img2 = np.squeeze(img2).astype(np.float32)

    if img1.shape != img2.shape:
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]

    max_val = max(img1.max(), img2.max())
    if max_val > 1.5:
        data_range = 255.0

    channel_axis = None
    if img1.ndim == 3 and img1.shape[-1] in (3, 4):
        channel_axis = -1
    elif img1.ndim == 3 and img1.shape[0] in (3, 4):
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
        channel_axis = -1

    return float(structural_similarity(img2, img1, data_range=data_range, channel_axis=channel_axis))


def calculate_lpips(
    img1: Union[np.ndarray, torch.Tensor],
    img2: Union[np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None
) -> Optional[float]:
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity).
    Expects inputs in range [0, 1].
    Gracefully returns None if lpips package is not installed or network weights unavailable.
    """
    global _LPIPS_MODEL, _LPIPS_AVAILABLE

    if not _LPIPS_AVAILABLE:
        return None

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _LPIPS_MODEL is None:
        try:
            import lpips
            _LPIPS_MODEL = lpips.LPIPS(net='alex', verbose=False).to(device)
            _LPIPS_MODEL.eval()
        except Exception as e:
            print(f"[Metrics] Warning: Could not initialize LPIPS ({e}). LPIPS evaluation will be skipped.")
            _LPIPS_AVAILABLE = False
            return None

    try:
        # Convert to numpy float32 [0, 1]
        if isinstance(img1, torch.Tensor):
            arr1 = img1.detach().cpu().squeeze().numpy().astype(np.float32)
        else:
            arr1 = np.squeeze(img1).astype(np.float32)

        if isinstance(img2, torch.Tensor):
            arr2 = img2.detach().cpu().squeeze().numpy().astype(np.float32)
        else:
            arr2 = np.squeeze(img2).astype(np.float32)

        if arr1.shape != arr2.shape:
            min_h = min(arr1.shape[0], arr2.shape[0])
            min_w = min(arr1.shape[1], arr2.shape[1])
            arr1 = arr1[:min_h, :min_w]
            arr2 = arr2[:min_h, :min_w]

        # Convert grayscale or RGB to (1, 3, H, W) normalized to [-1, 1]
        def to_lpips_tensor(arr):
            if arr.ndim == 2:
                t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            elif arr.ndim == 3 and arr.shape[-1] == 3:
                t = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0)
            elif arr.ndim == 3 and arr.shape[0] == 3:
                t = torch.from_numpy(arr).unsqueeze(0)
            else:
                t = torch.from_numpy(arr).unsqueeze(0).repeat(1, 3, 1, 1)
            # Normalize [0, 1] -> [-1, 1]
            return (t.float().to(device) * 2.0) - 1.0

        t1 = to_lpips_tensor(arr1)
        t2 = to_lpips_tensor(arr2)

        with torch.no_grad():
            lpips_val = _LPIPS_MODEL(t1, t2).item()
        return float(lpips_val)
    except Exception as e:
        print(f"[Metrics] Warning: LPIPS computation error ({e}).")
        return None


def ssim_torch(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    """
    Differentiable SSIM calculation in PyTorch for training loss.
    Expects input tensors in range [0, 1] with shape (B, C, H, W).
    """
    def gaussian(w_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - w_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(w_size)])
        return gauss / gauss.sum()

    def create_window(w_size, channel):
        _1D_window = gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window

    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
