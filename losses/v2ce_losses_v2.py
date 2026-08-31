import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.v2ce_losses import (
    OccupancyWeightedEFLoss,
    SpatialSpectralLoss,
    TemporalSpectralLoss,
)


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
    ) -> None:
        super().__init__()
        self.lambda_stp = lambda_stp
        self.lambda_tp  = lambda_tp
        self.lambda_ef  = lambda_ef
        self.lambda_ss  = lambda_ss
        self.lambda_ts  = lambda_ts
        self.stp = STPMMLoss()
        self.tp  = TPMMLoss()
        self.ef  = OccupancyWeightedEFLoss()
        self.ss  = SpatialSpectralLoss()
        self.ts  = TemporalSpectralLoss()

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # pred, gt: (B, T, H, W)
        stp_val             = self.stp(pred, gt)
        tp_val              = self.tp(pred, gt)
        ef_val, _ = self.ef(pred, gt)
        ss_val    = self.ss(pred, gt)
        ts_val    = self.ts(pred, gt)
        total = (
            self.lambda_stp * stp_val
            + self.lambda_tp  * tp_val
            + self.lambda_ef  * ef_val
            + self.lambda_ss  * ss_val
            + self.lambda_ts  * ts_val
        )
        return {
            "total": total,
            "stp":   stp_val,
            "tp":    tp_val,
            "ef":    ef_val,
            "ss":    ss_val,
            "ts":    ts_val,
        }
