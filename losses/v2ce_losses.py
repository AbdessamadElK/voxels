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

    def forward(self, _pred: torch.Tensor, _gt: torch.Tensor) -> torch.Tensor:
        # deactivated — replaced by pyramid moment matching, see v2ce_losses_v2.py.
        # pred5 = _pred.unsqueeze(1)   # (B, 1, T, H, W)
        # gt5   = gt.unsqueeze(1)     # (B, 1, T, H, W)
        # loss = pred.new_zeros(1).squeeze()
        # for k, s in self.pyramid_levels:
        #     p = F.avg_pool3d(pred5, kernel_size=k, stride=s, padding=0)
        #     g = F.avg_pool3d(gt5,   kernel_size=k, stride=s, padding=0)
        #     loss = loss + F.l1_loss(p, g)
        # return loss
        raise NotImplementedError


class TPLoss(nn.Module):
    """Temporal-Pyramid loss: multi-scale 1D pooling of spatial-collapsed profiles.

    Collapses H and W into a temporal profile, then applies 1D average pooling at
    multiple scales and sums L1 across all scales.

    The legacy spatial_reduce="sum" multiplied every profile value by H×W (~9e4 at
    346×260), inflating the loss into the tens of thousands. The default
    spatial_reduce="mean" keeps values in per-pixel, resolution-independent units,
    matching the scale of STPLoss and EFLoss (~2–6).

    Set normalize_profiles=True to compare temporal shape only: profiles are divided
    by their total mass before pooling, so absolute event count is ignored.

    Args:
        temporal_scales: kernel sizes for F.avg_pool1d.
        spatial_reduce: "mean" (default, resolution-independent) or "sum" (legacy).
        normalize_profiles: divide each profile by its total mass before comparison.
        eps: small constant to avoid division by zero in profile normalization.
    """

    def __init__(
        self,
        temporal_scales: list[int] = None,
        spatial_reduce: str = "mean",
        normalize_profiles: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if spatial_reduce not in ("mean", "sum"):
            raise ValueError(f"spatial_reduce must be 'mean' or 'sum', got {spatial_reduce!r}")
        if temporal_scales is None:
            temporal_scales = [1, 2, 4, 8]
        self.temporal_scales = temporal_scales
        self.spatial_reduce = spatial_reduce
        self.normalize_profiles = normalize_profiles
        self.eps = eps

    def _profile(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W) -> (B, T)
        if self.spatial_reduce == "mean":
            p = x.mean(dim=(-2, -1))
        else:
            p = x.sum(dim=(-2, -1))
        if self.normalize_profiles:
            p = p / p.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return p

    def forward(self, _pred: torch.Tensor, _gt: torch.Tensor) -> torch.Tensor:
        # deactivated — replaced by pyramid moment matching, see v2ce_losses_v2.py.
        # pred_seq = self._profile(_pred).unsqueeze(1)  # (B, 1, T)
        # gt_seq   = self._profile(_gt).unsqueeze(1)    # (B, 1, T)
        # T = pred_seq.shape[-1]
        # loss = _pred.new_zeros(1).squeeze()
        # for k in self.temporal_scales:
        #     if k > T:
        #         continue
        #     p = F.avg_pool1d(pred_seq, kernel_size=k, stride=k, padding=0)
        #     g = F.avg_pool1d(gt_seq,   kernel_size=k, stride=k, padding=0)
        #     loss = loss + F.l1_loss(p, g)
        # return loss
        raise NotImplementedError


class EFLoss(nn.Module):
    """Event-Frame loss: L1 on event frames produced by summing over the time axis."""

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # pred, gt: (B, T, H, W)
        pred_frame = pred.sum(dim=1)    # (B, H, W)
        gt_frame   = gt.sum(dim=1)      # (B, H, W)
        return F.l1_loss(pred_frame, gt_frame)


class OccupancyWeightedEFLoss(nn.Module):
    """Event-frame L1 with separate foreground / background region means.

    Accumulates pred and gt over T, builds a binary occupancy mask from the GT
    absolute sum, then computes per-region mean absolute errors and returns
    fg_mean + beta * bg_mean averaged over the batch.

    Decoupling the two region means prevents the empty-pixel majority from
    dominating: with ~19 background pixels per active pixel a single global
    average buries the foreground signal. Separate means make the active-pixel
    error the primary signal regardless of sparsity. beta is a light leash on
    the background; raise it if false positives appear, set to 0 to ignore
    background entirely.

    Args:
        beta: weight on the background region mean.
        thr:  occupancy threshold on gt.abs().sum(dim=1); pixels above are foreground.
        eps:  floor for region pixel counts to prevent division by zero.
    """

    def __init__(self, beta: float = 0.1, thr: float = 0.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.beta = beta
        self.thr  = thr
        self.eps  = eps

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # pred, gt: (B, T, H, W)
        pred_frame = pred.sum(dim=1)                        # (B, H, W)
        gt_frame   = gt.sum(dim=1)                          # (B, H, W)

        # occupancy mask from GT — no gradient flows through it
        with torch.no_grad():
            occ = (gt.abs().sum(dim=1) > self.thr).float() # (B, H, W)

        diff = (pred_frame - gt_frame).abs()                # (B, H, W)
        inv  = 1.0 - occ

        # per-sample region means, then average over batch
        fg = (diff * occ).sum(dim=(-2, -1)) / occ.sum(dim=(-2, -1)).clamp_min(self.eps)
        bg = (diff * inv).sum(dim=(-2, -1)) / inv.sum(dim=(-2, -1)).clamp_min(self.eps)

        loss = (fg + self.beta * bg).mean()

        monitors = {
            "ef_fg":        fg.mean().detach(),
            "ef_bg":        bg.mean().detach(),
            "active_frac":  occ.mean().detach(),
        }
        return loss, monitors


class SpatialSpectralLoss(nn.Module):
    """L1 loss on log-magnitude 2-D spectra of accumulated event frames.

    Sums pred and gt over T, applies rfft2, takes log(|F| + 1e-8), and returns
    plain F.l1_loss over all bins including DC.
    """

    EPS: float = 1e-8

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # pred, gt: (B, T, H, W) -> sum over T -> (B, H, W)
        # cast to float32: cuFFT half-precision requires power-of-2 spatial dims
        pred_frame = pred.sum(dim=1).float()  # (B, H, W)
        gt_frame   = gt.sum(dim=1).float()    # (B, H, W)

        pred_spec = torch.fft.rfft2(pred_frame)  # (B, H, W//2+1) complex
        gt_spec   = torch.fft.rfft2(gt_frame)    # (B, H, W//2+1) complex

        pred_log = torch.log(pred_spec.abs() + self.EPS)  # (B, H, W//2+1)
        gt_log   = torch.log(gt_spec.abs()   + self.EPS)  # (B, H, W//2+1)

        return F.l1_loss(pred_log, gt_log)


class TemporalSpectralLoss(nn.Module):
    """L1 loss on spatially-averaged log-magnitude temporal power profiles.

    Targets the temporal rhythm of event activity per spatial location, averaged
    over H and W. Penalizes wrong oscillation rates and uniform or jittery temporal
    profiles, independently of spatial layout.

    Args:
        eps: small constant added before log for numerical stability.
    """

    EPS: float = 1e-8

    def __init__(self, eps: float = EPS) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # pred, gt: (B, T, H, W) -> (B, H, W, T)
        # cast to float32: cuFFT half-precision requires power-of-2 T
        pred_t = pred.float().permute(0, 2, 3, 1)
        gt_t   = gt.float().permute(0, 2, 3, 1)
        # (B, H, W, T//2+1) complex
        pred_spec = torch.fft.rfft(pred_t, dim=-1)
        gt_spec   = torch.fft.rfft(gt_t,   dim=-1)
        pred_log  = torch.log(pred_spec.abs() + self.eps)   # (B, H, W, T//2+1)
        gt_log    = torch.log(gt_spec.abs()   + self.eps)
        # average over H and W -> (B, T//2+1)
        pred_profile = pred_log.mean(dim=(1, 2))
        gt_profile   = gt_log.mean(dim=(1, 2))
        return F.l1_loss(pred_profile, gt_profile)


class CombinedLoss(nn.Module):
    """Weighted sum of STPLoss, TPLoss, EFLoss, SpatialSpectralLoss, and TemporalSpectralLoss.

    Returns a dict with keys 'total', 'stp', 'tp', 'ef', 'ss', 'ts'.
    """

    def __init__(
        self,
        lambda_stp: float = 1.0,
        lambda_tp: float = 1.0,
        lambda_ef: float = 1.0,
        lambda_ss: float = 1.0,
        lambda_ts: float = 1.0,
        tp_kwargs: dict = None,
    ) -> None:
        super().__init__()
        self.lambda_stp = lambda_stp
        self.lambda_tp  = lambda_tp
        self.lambda_ef  = lambda_ef
        self.lambda_ss  = lambda_ss
        self.lambda_ts  = lambda_ts
        self.stp = STPLoss()
        self.tp  = TPLoss(**(tp_kwargs or {}))
        self.ef  = EFLoss()
        self.ss  = SpatialSpectralLoss()
        self.ts  = TemporalSpectralLoss()

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # pred, gt: (B, T, H, W)
        stp_val = self.stp(pred, gt)
        tp_val  = self.tp(pred, gt)
        ef_val  = self.ef(pred, gt)
        ss_val  = self.ss(pred, gt)
        ts_val  = self.ts(pred, gt)
        total = (
            self.lambda_stp * stp_val
            + self.lambda_tp  * tp_val
            + self.lambda_ef  * ef_val
            + self.lambda_ss  * ss_val
            + self.lambda_ts  * ts_val
        )
        return {"total": total, "stp": stp_val, "tp": tp_val, "ef": ef_val, "ss": ss_val, "ts": ts_val}
