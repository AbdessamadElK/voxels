import torch
import torch.nn as nn
import torch.nn.functional as F

# Ring index maps depend only on geometry, so cache them per (H, W, device).
_RADIUS_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

DEFAULT_EPS = 1e-8


def _ring_index(H: int, W: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the flattened integer radius per rfft2 bin and the per-radius bin count."""
    key = (H, W, str(device))
    if key not in _RADIUS_CACHE:
        fy = torch.arange(H, dtype=torch.float32, device=device)
        fy = torch.where(fy <= H // 2, fy, fy - H)          # centre DC: [-H/2, H/2)
        fx = torch.arange(W // 2 + 1, dtype=torch.float32, device=device)
        gy, gx = torch.meshgrid(fy, fx, indexing="ij")      # (H, W//2+1)
        radius = (gy ** 2 + gx ** 2).sqrt().round().long().flatten()
        counts = torch.bincount(radius).float().clamp_min(1.0)
        _RADIUS_CACHE[key] = (radius, counts)
    return _RADIUS_CACHE[key]


class RAPSLoss(nn.Module):
    """L1 on the log radially-averaged power spectrum of accumulated event frames.

    Sums pred and gt over T, takes rfft2, bins |F|^2 into integer-radius rings, and
    compares the log of the per-ring means. Constrains the radial frequency profile —
    how energy distributes from low to high frequency — without pinning individual
    bins, which is what separates it from SpatialSpectralLoss.

    This mirrors `training.metrics._raps`. Optimising it makes RAPS a training
    objective rather than an independent monitor; validate on something else.

    Args:
        eps: floor inside the log. Ring means sum many bins, so they rarely approach
            zero and the default is safe here — unlike the per-pixel temporal spectrum,
            where a small eps put 1/(x+eps) on empty bins. See NOTES.md Session 9.
    """

    def __init__(self, eps: float = DEFAULT_EPS) -> None:
        super().__init__()
        self.eps = eps

    def _profile(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, H, W) -> (B, num_rings)
        B, _, H, W = x.shape
        frame = x.sum(dim=1).float()                        # (B, H, W)
        power = torch.fft.rfft2(frame).abs().pow(2)         # (B, H, W//2+1)

        radius, counts = _ring_index(H, W, x.device)
        idx = radius.unsqueeze(0).expand(B, -1)             # (B, H*(W//2+1))
        rings = power.flatten(1).new_zeros(B, counts.numel())
        rings = rings.scatter_add(1, idx, power.flatten(1))
        return rings / counts

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # pred, gt: (B, T, H, W)
        return F.l1_loss(
            torch.log(self._profile(pred) + self.eps),
            torch.log(self._profile(gt) + self.eps),
        )
