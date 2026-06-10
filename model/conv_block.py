import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Residual 3x3 conv block: two Conv2d+GELU with a skip connection."""

    def __init__(self, dim: int, bias: bool = False) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)  # (B, C, H, W) -> (B, C, H, W)
