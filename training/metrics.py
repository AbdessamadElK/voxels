import torch
import torch.nn.functional as F

try:
    from pytorch_msssim import ssim as _msssim
    _HAS_MSSSIM = True
except ImportError:
    _HAS_MSSSIM = False

_EPS = 1e-8
_RADIUS_CACHE: dict[tuple, torch.Tensor] = {}
_KERNEL_CACHE: dict[tuple, torch.Tensor] = {}


# ---------------------------------------------------------------------------
# RAPS — Radially Averaged Power Spectrum L1
# ---------------------------------------------------------------------------

def _radius_map(H: int, W: int, device: torch.device) -> torch.Tensor:
    key = (H, W, str(device))
    if key not in _RADIUS_CACHE:
        fy = torch.arange(H, dtype=torch.float32, device=device)
        fy = torch.where(fy <= H // 2, fy, fy - H)       # center DC: [-H/2, H/2)
        fx = torch.arange(W // 2 + 1, dtype=torch.float32, device=device)
        gy, gx = torch.meshgrid(fy, fx, indexing="ij")    # (H, W//2+1)
        _RADIUS_CACHE[key] = (gy ** 2 + gx ** 2).sqrt().round().long()
    return _RADIUS_CACHE[key]


def _raps(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    # pred, gt: (B, T, H, W)
    B, _, H, W = pred.shape
    pf = pred.sum(1).float()                               # (B, H, W)
    gf = gt.sum(1).float()

    pp = torch.fft.rfft2(pf).abs().pow(2)                  # (B, H, W//2+1)
    gp = torch.fft.rfft2(gf).abs().pow(2)

    rmap   = _radius_map(H, W, pred.device)                # (H, W//2+1)
    max_r  = int(rmap.max().item()) + 1
    flat_r = rmap.flatten()                                 # (H*(W//2+1),)

    idx    = flat_r.unsqueeze(0).expand(B, -1)             # (B, H*(W//2+1))
    pp_ring = pp.flatten(1).new_zeros(B, max_r)
    gp_ring = gp.flatten(1).new_zeros(B, max_r)
    pp_ring.scatter_add_(1, idx, pp.flatten(1))
    gp_ring.scatter_add_(1, idx, gp.flatten(1))

    counts = torch.bincount(flat_r, minlength=max_r).float().clamp_min_(1.0)
    pp_ring = pp_ring / counts
    gp_ring = gp_ring / counts

    return F.l1_loss(torch.log(pp_ring + _EPS), torch.log(gp_ring + _EPS))


# ---------------------------------------------------------------------------
# SSIM — Gaussian-window, data_range from per-sample GT max
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    key = (size, str(device))
    if key not in _KERNEL_CACHE:
        c = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        g = torch.exp(-(c ** 2) / (2.0 * sigma ** 2))
        g = g / g.sum()
        _KERNEL_CACHE[key] = g.outer(g).unsqueeze(0).unsqueeze(0)  # (1,1,size,size)
    return _KERNEL_CACHE[key]


def _ssim_pair(x: torch.Tensor, y: torch.Tensor, data_range: float) -> torch.Tensor:
    # x, y: (1, 1, H, W)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    k  = _gaussian_kernel(11, 1.5, x.device)
    mu_x  = F.conv2d(x,     k, padding=5)
    mu_y  = F.conv2d(y,     k, padding=5)
    mu_xx = mu_x.pow(2)
    mu_yy = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sg_xx = F.conv2d(x * x, k, padding=5) - mu_xx
    sg_yy = F.conv2d(y * y, k, padding=5) - mu_yy
    sg_xy = F.conv2d(x * y, k, padding=5) - mu_xy
    num   = (2.0 * mu_xy + C1) * (2.0 * sg_xy + C2)
    den   = (mu_xx + mu_yy + C1) * (sg_xx + sg_yy + C2)
    return (num / den).mean()


def _ssim(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    # pred, gt: (B, 1, H, W), float32
    vals = []
    for b in range(pred.size(0)):
        dr = max(float(gt[b].max().item()), _EPS)
        if _HAS_MSSSIM:
            vals.append(_msssim(pred[b:b+1], gt[b:b+1], data_range=dr, size_average=True))
        else:
            vals.append(_ssim_pair(pred[b:b+1], gt[b:b+1], dr))
    return torch.stack(vals).mean()


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    """Return {"raps": float, "ssim": float}, averaged over the batch.

    pred, gt: (B, T, H, W). Tensors may be float16; cast to float32 internally.
    """
    raps_val = _raps(pred, gt)
    pf = pred.sum(1, keepdim=True).float()   # (B, 1, H, W)
    gf = gt.sum(1, keepdim=True).float()
    ssim_val = _ssim(pf, gf)
    return {"raps": float(raps_val), "ssim": float(ssim_val)}
