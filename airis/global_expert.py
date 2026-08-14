import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttentionBlock(nn.Module):
    """
    Window-based Multi-Head Self-Attention Block for efficient global context.
    Partitions feature maps into (window_size x window_size) non-overlapping windows.
    """
    def __init__(self, dim: int = 48, window_size: int = 8, num_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, C, H, W)
        Output: (B, C, H, W)
        """
        B, C, H, W = x.shape
        ws = self.window_size

        # Pad if H or W not divisible by window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        _, _, Hp, Wp = x.shape

        # Permute: (B, C, Hp, Wp) -> (B, Hp, Wp, C)
        x_perm = x.permute(0, 2, 3, 1).contiguous()

        # Partition into windows: (B * num_windows, window_size * window_size, C)
        num_h = Hp // ws
        num_w = Wp // ws
        windows = x_perm.view(B, num_h, ws, num_w, ws, C)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)

        # 1. Multihead Attention with residual
        norm_windows = self.norm1(windows)
        attn_out, _ = self.attn(norm_windows, norm_windows, norm_windows)
        windows = windows + attn_out

        # 2. MLP with residual
        windows = windows + self.mlp(self.norm2(windows))

        # Reverse window partition: -> (B, Hp, Wp, C)
        x_rev = windows.view(B, num_h, num_w, ws, ws, C)
        x_rev = x_rev.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)

        # Permute back: (B, C, Hp, Wp)
        out = x_rev.permute(0, 3, 1, 2).contiguous()

        # Crop back to original dimensions if padded
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return out


class GlobalContextExpert(nn.Module):
    """
    Global Context Expert for AIRIS-Net.
    Captures long-range structural dependencies and geometric relationships using Window-based Transformers.
    """
    def __init__(self, channels: int = 48, window_size: int = 8, num_heads: int = 4, depth: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([
            WindowAttentionBlock(dim=channels, window_size=window_size, num_heads=num_heads)
            for _ in range(depth)
        ])
        self.proj = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: F0 (B, channels, H, W)
        Output: F_global (B, channels, H, W)
        """
        res = x
        out = x
        for block in self.blocks:
            out = block(out)
        out = self.proj(out)
        return res + out
