import torch
import torch.nn as nn
import torch.nn.functional as F


class STPLoss(nn.Module):
    """Spatial-Temporal-Pyramid loss: multi-scale 3D average pooling L1.

    Pools predicted and ground-truth voxels at increasing spatial-temporal scales
    and sums L1 across all pyramid levels.

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
            loss = loss + F.l1_loss(p, g)
        return loss


class TPLoss(nn.Module):
    """Temporal-Pyramid loss: multi-scale 1D pooling of spatial-collapsed profiles.

    Collapses H and W by summing, then applies 1D average pooling at multiple scales
    and sums L1 across all scales.

    Args:
        temporal_scales: list of kernel sizes for F.avg_pool1d.
    """

    def __init__(self, temporal_scales: list[int] = None) -> None:
        super().__init__()
        if temporal_scales is None:
            temporal_scales = [1, 2, 4, 8, 16]
        self.temporal_scales = temporal_scales

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # pred, gt: (B, T, H, W)
        pred_profile = pred.sum(dim=(-2, -1))   # (B, T)
        gt_profile   = gt.sum(dim=(-2, -1))     # (B, T)
        pred_seq = pred_profile.unsqueeze(1)     # (B, 1, T)
        gt_seq   = gt_profile.unsqueeze(1)       # (B, 1, T)
        loss = pred.new_zeros(1).squeeze()
        for k in self.temporal_scales:
            p = F.avg_pool1d(pred_seq, kernel_size=k, stride=k, padding=0)
            g = F.avg_pool1d(gt_seq,   kernel_size=k, stride=k, padding=0)
            loss = loss + F.l1_loss(p, g)
        return loss


class EFLoss(nn.Module):
    """Event-Frame loss: L1 on event frames produced by summing over the time axis."""

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # pred, gt: (B, T, H, W)
        pred_frame = pred.sum(dim=1)    # (B, H, W)
        gt_frame   = gt.sum(dim=1)      # (B, H, W)
        return F.l1_loss(pred_frame, gt_frame)


class CombinedLoss(nn.Module):
    """Weighted sum of STPLoss, TPLoss, and EFLoss.

    Returns a dict with keys 'total', 'stp', 'tp', 'ef'.
    """

    def __init__(
        self,
        lambda_stp: float = 1.0,
        lambda_tp: float = 1.0,
        lambda_ef: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_stp = lambda_stp
        self.lambda_tp  = lambda_tp
        self.lambda_ef  = lambda_ef
        self.stp = STPLoss()
        self.tp  = TPLoss()
        self.ef  = EFLoss()

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # pred, gt: (B, T, H, W)
        stp_val = self.stp(pred, gt)
        tp_val  = self.tp(pred, gt)
        ef_val  = self.ef(pred, gt)
        total   = self.lambda_stp * stp_val + self.lambda_tp * tp_val + self.lambda_ef * ef_val
        return {"total": total, "stp": stp_val, "tp": tp_val, "ef": ef_val}
