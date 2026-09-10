import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.raps_loss import RAPSLoss
from losses.v2ce_losses import (
    OccupancyWeightedEFLoss,
    SpatialSpectralLoss,
    TemporalSpectralLoss,
)

# TemporalSpectralLoss takes log(|F| + eps). Every run up to 2026-09-08 used 1e-8,
# and this stays the constructor default so those results remain reproducible — set
# ts_eps from the loss config to change it. 1e-8 is far below the GT spectrum floor
# (p10 = 0.51 on event pixels) while ~45% of bins are exactly zero, so d/dx
# log(x + eps) = 1/eps = 1e8 lands on empty bins: the ts gradient came out 16x the
# other four terms at init and 149x by step 2500. See NOTES.md Session 9.
LEGACY_TS_EPS = 1e-8


def _moments(x: torch.Tensor) -> torch.Tensor:
    """Return (B, 3) tensor of mean, standard deviation, and skewness per sample."""
    x_flat = x.flatten(start_dim=1)                            # (B, N)
    mu     = x_flat.mean(dim=1)                                # (B,)
    x_c    = x_flat - mu.unsqueeze(1)                          # (B, N)
    var    = (x_c ** 2).mean(dim=1)                            # (B,)
    std    = (var + 1e-8).sqrt()                               # (B,)
    skew   = (x_c ** 3).mean(dim=1) / (std ** 3 + 1e-8)       # (B,)
    return torch.stack([mu, std, skew], dim=1)                  # (B, 3)


class STPMMLoss(nn.Module):
    """Pyramid moment matching over 3-D spatial-temporal scales.

    Pools pred and gt voxels with F.avg_pool3d at each pyramid level, then matches
    the distribution of pooled activations via mean, standard deviation, and skewness.
    Penalizes distributional mismatch at every scale without requiring element-wise
    spatial alignment.

    Args:
        pyramid_levels: list of (kernel_size, stride) for F.avg_pool3d.
    """

    def __init__(
        self,
        pyramid_levels: list[tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        if pyramid_levels is None:
            pyramid_levels = [(1, 1), (2, 2), (4, 4), (8, 8)]
        self.pyramid_levels = pyramid_levels

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # pred, gt: (B, T, H, W)
        pred5 = pred.unsqueeze(1)   # (B, 1, T, H, W)
        gt5   = gt.unsqueeze(1)     # (B, 1, T, H, W)
        loss = pred.new_zeros(1).squeeze()
        for k, s in self.pyramid_levels:
            p = F.avg_pool3d(pred5, kernel_size=k, stride=s, padding=0)
            g = F.avg_pool3d(gt5,   kernel_size=k, stride=s, padding=0)
            loss = loss + F.l1_loss(_moments(p), _moments(g))
        return loss


class TPMMLoss(nn.Module):
    """Pyramid moment matching over temporal activity profiles.

    Collapses H and W by averaging to a per-timestep activity profile (B, T), then
    applies F.avg_pool1d at each temporal scale and matches the distribution of
    pooled profiles via mean, standard deviation, and skewness. Penalizes wrong
    oscillation rates and distributional shape without requiring frame-level alignment.

    Args:
        temporal_scales: kernel sizes for F.avg_pool1d.
    """

    def __init__(
        self,
        temporal_scales: list[int] = None,
    ) -> None:
        super().__init__()
        if temporal_scales is None:
            temporal_scales = [1, 2, 4, 8, 16]
        self.temporal_scales = temporal_scales

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # pred, gt: (B, T, H, W) -> sum H,W -> (B, T) -> (B, 1, T)
        pred_seq = pred.mean(dim=(-2, -1)).unsqueeze(1)
        gt_seq   = gt.mean(dim=(-2, -1)).unsqueeze(1)
        T = pred_seq.shape[-1]
        loss = pred.new_zeros(1).squeeze()
        for k in self.temporal_scales:
            if k > T:
                continue
            p = F.avg_pool1d(pred_seq, kernel_size=k, stride=k, padding=0)
            g = F.avg_pool1d(gt_seq,   kernel_size=k, stride=k, padding=0)
            loss = loss + F.l1_loss(_moments(p), _moments(g))
        return loss


class CombinedLoss(nn.Module):
    """Weighted sum of STPMMLoss, TPMMLoss, OccupancyWeightedEFLoss, SpatialSpectralLoss, and TemporalSpectralLoss.

    Returns a dict with keys 'total', 'stp', 'tp', 'ef', 'ss', 'ts',
    plus monitors 'ef_fg', 'ef_bg', 'active_frac', 'ef_plain'.
    """

    def __init__(
        self,
        lambda_stp: float = 1.0,
        lambda_tp:  float = 1.0,
        lambda_ef:  float = 1.0,
        lambda_ss:  float = 1.0,
        lambda_ts:  float = 1.0,
        lambda_raps: float = 0.0,
        ts_eps:     float = LEGACY_TS_EPS,
    ) -> None:
        super().__init__()
        self.lambda_stp = lambda_stp
        self.lambda_tp  = lambda_tp
        self.lambda_ef  = lambda_ef
        self.lambda_ss  = lambda_ss
        self.lambda_ts  = lambda_ts
        self.lambda_raps = lambda_raps
        self.ts_eps     = ts_eps
        self.stp = STPMMLoss()
        self.tp  = TPMMLoss()
        self.ef  = OccupancyWeightedEFLoss()
        self.ss  = SpatialSpectralLoss()
        self.ts  = TemporalSpectralLoss(eps=ts_eps)
        self.raps = RAPSLoss()

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # pred, gt: (B, T, H, W)
        stp_val             = self.stp(pred, gt)
        tp_val              = self.tp(pred, gt)
        ef_val, _ = self.ef(pred, gt)
        ss_val    = self.ss(pred, gt)
        ts_val    = self.ts(pred, gt)
        # Skip the FFT when the term is switched off (lambda_raps defaults to 0).
        raps_val = (
            self.raps(pred, gt)
            if self.lambda_raps
            else pred.new_zeros(1).squeeze()
        )
        total = (
            self.lambda_stp * stp_val
            + self.lambda_tp  * tp_val
            + self.lambda_ef  * ef_val
            + self.lambda_ss  * ss_val
            + self.lambda_ts  * ts_val
            + self.lambda_raps * raps_val
        )
        return {
            "total": total,
            "stp":   stp_val,
            "tp":    tp_val,
            "ef":    ef_val,
            "ss":    ss_val,
            "ts":    ts_val,
            "raps":  raps_val,
        }
