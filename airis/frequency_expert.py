import torch
import torch.nn as nn
import torch.fft
from typing import Tuple


class FrequencyExpert(nn.Module):
    """
    Frequency Domain Expert for AIRIS-Net.
    Decomposes feature maps into Low, Mid, and High frequency bands using 2D FFT,
    processes each band with dedicated learnable operations, and recombines them.
    Fully differentiable end-to-end.
    """
    def __init__(self, channels: int = 48):
        super().__init__()
        self.channels = channels

        # Dedicated lightweight processing for each frequency band
        self.proc_low = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GELU()
        )
        self.proc_mid = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GELU()
        )
        self.proc_high = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GELU()
        )

        # 1x1 fusion to combine all 3 processed bands
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        )

    def _generate_frequency_masks(self, H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate radial frequency masks for Low, Mid, and High frequency bands.
        """
        y = torch.linspace(-1.0, 1.0, steps=H, device=device)
        x = torch.linspace(-1.0, 1.0, steps=W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        radius = torch.sqrt(xx ** 2 + yy ** 2)  # 0.0 at center, ~1.414 at corners

        # Normalized radial thresholds
        low_mask = (radius <= 0.33).float().unsqueeze(0).unsqueeze(0)
        mid_mask = ((radius > 0.33) & (radius <= 0.66)).float().unsqueeze(0).unsqueeze(0)
        high_mask = (radius > 0.66).float().unsqueeze(0).unsqueeze(0)

        return low_mask, mid_mask, high_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: F0 (B, C, H, W)
        Output: F_frequency (B, C, H, W)
        """
        B, C, H, W = x.shape
        device = x.device

        # 1. 2D FFT to frequency domain
        fft_feat = torch.fft.fft2(x, norm='ortho')
        fft_shift = torch.fft.fftshift(fft_feat, dim=(-2, -1))

        # 2. Extract Low, Mid, High bands using radial masks
        low_mask, mid_mask, high_mask = self._generate_frequency_masks(H, W, device)

        fft_low = fft_shift * low_mask
        fft_mid = fft_shift * mid_mask
        fft_high = fft_shift * high_mask

        # 3. Inverse FFT back to spatial domain
        low_spatial = torch.fft.ifft2(torch.fft.ifftshift(fft_low, dim=(-2, -1)), norm='ortho').real
        mid_spatial = torch.fft.ifft2(torch.fft.ifftshift(fft_mid, dim=(-2, -1)), norm='ortho').real
        high_spatial = torch.fft.ifft2(torch.fft.ifftshift(fft_high, dim=(-2, -1)), norm='ortho').real

        # 4. Bandwise learnable operations
        f_low = self.proc_low(low_spatial)
        f_mid = self.proc_mid(mid_spatial)
        f_high = self.proc_high(high_spatial)

        # 5. Concatenate and fuse
        f_concat = torch.cat([f_low, f_mid, f_high], dim=1)
        out = self.fusion(f_concat)

        return x + out
