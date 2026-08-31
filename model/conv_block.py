import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Residual depthwise-separable conv block: (DW 3×3 → PW 1×1 → GELU) × 2 with skip."""

    def __init__(self, dim: int, bias: bool = False) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias),  # DW 3×3
            nn.Conv2d(dim, dim, 1, bias=bias),                          # PW 1×1
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias),  # DW 3×3
            nn.Conv2d(dim, dim, 1, bias=bias),                          # PW 1×1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)  # (B, C, H, W) -> (B, C, H, W)
