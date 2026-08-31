# Project Notes

## Loss Architecture

### STPLoss
Multi-scale 3D average-pool L1. Operates on `(B, 1, T, H, W)` through `F.avg_pool3d` at four pyramid levels `[(1,1),(2,2),(4,4),(8,8)]`. Returns the sum of per-level L1 values, each averaged over all elements by `F.l1_loss`. Typical range with real voxel data: 2–6.

### TPLoss
Temporal-profile L1 with multi-scale 1D pooling. Collapses spatial dimensions to a `(B, T)` profile, wraps it as `(B, 1, T)`, then applies `F.avg_pool1d` at scales `[1, 2, 4, 8]` and sums L1 across scales.

**Parameters:**
- `temporal_scales` — pool kernel sizes (default `[1, 2, 4, 8]`)
- `spatial_reduce` — `"mean"` (default) or `"sum"` (legacy)
- `normalize_profiles` — divide profile by its total mass before comparison; compares temporal shape only
- `eps` — floor for normalization denominator (default `1e-6`)

### EFLoss
L1 on event frames: sums the voxel grid over the time axis to produce `(B, H, W)` frames, then `F.l1_loss`. Typical range: 2–6.

### CombinedLoss
Weighted sum `lambda_stp * stp + lambda_tp * tp + lambda_ef * ef`. Returns `{"total", "stp", "tp", "ef"}`.

**Constructor:**
```python
CombinedLoss(
    lambda_stp=1.0,
    lambda_tp=1.0,
    lambda_ef=1.0,
    tp_kwargs={"spatial_reduce": "mean"},  # or any TPLoss kwargs
)
```

---

## Lambda Calibration Recipe

Forward a few hundred real batches with all lambdas at 1. Track running means of each raw component loss. Then set:

```
lambda_i = target_scale / mean(loss_i)
```

where `target_scale` is a common reference (e.g. the mean of `stp` and `ef`). Re-check balance mid-training: components decay at different rates and the ratios shift.

---

## Session Log

### 2026-06-11 — TPLoss scale blow-up fix

**Diagnosis.** `TPLoss` converged at values above 10 000 while `STPLoss` and `EFLoss` stayed in the 2–6 range. Root cause: `pred.sum(dim=(-2, -1))` in `forward` multiplied every profile value by `H × W ≈ 89 960` at 346 × 260 resolution. `F.l1_loss` averages over elements, so the resulting L1 was inflated by that factor. `STPLoss` and `EFLoss` both reduce spatially through averaging, so they were unaffected.

**Fix.** Added `spatial_reduce: str = "mean"` (default), `normalize_profiles: bool = False`, and `eps: float = 1e-6` to `TPLoss.__init__`. Extracted a `_profile` method that dispatches between `mean` and `sum` reduction and optionally normalises by total mass. `forward` now calls `_profile` instead of inlining the sum. `CombinedLoss` gained a `tp_kwargs: dict` parameter so callers can pass any `TPLoss` option without subclassing.

**Verification** (random voxels `(2, 16, 260, 346)`):
```
stp  = 0.5037
ef   = 1.3015
tp (legacy sum)    = 204.3857   # H×W inflated
tp (mean)          =   0.0023   # fixed; use lambda_tp to re-balance
tp (mean+norm)     =   0.0003   # shape-only
gradients finite: True
```

The `mean` variant eliminates the H×W inflation. Apply the lambda calibration recipe above to re-balance `tp` against `stp`/`ef` once real-data statistics are available.

---

## Session 2 — 2026-06-18 — Conditional GAN training path

### New files

| File | Role |
|---|---|
| `model/discriminator.py` | Three discriminator variants + `build_discriminator` factory |
| `losses/gan_loss.py` | `GANLoss` (4 modes) + standalone `gradient_penalty` |
| `training/trainer_gan.py` | `GANTrainer`, `GANTrainerConfig`, `VoxelPool` |
| `train_gan.py` | CLI entry point mirroring `train.py` |

### Discriminator design choices

**PatchGAN (`basic`) over a U-Net discriminator.** The generator is already a heavy UNet-Transformer; doubling the U-Net depth on the discriminator side would double VRAM at training time and add little signal — the reconstruction losses (`STPMMLoss`, `TPMMLoss`, spectral terms) already carry global structure. A 70-px PatchGAN is cheap, penalises local texture mismatch independently at each patch, and is the pix2pix baseline.

**`multiscale` available as an option.** When single-scale PatchGAN saturates early (discriminator always wins), `--netD multiscale` adds a second scale at half resolution with `--num_D 2`.

**Conditional input: 15 + 15 = 30 channels.** The discriminator receives `[cond, target]` concatenated on the channel dimension, so it always sees the synthetic input alongside the real/fake real-event voxel. This ties the adversarial signal to the conditioning context and prevents the discriminator from ignoring the generator input.

### GAN modes

| Mode | D loss | G loss | Notes |
|---|---|---|---|
| `lsgan` | MSE vs 1/0 | MSE vs 1 | Default; stable gradients |
| `vanilla` | BCE | BCE | Original GAN |
| `hinge` | relu(1−D(r)) + relu(1+D(f)) | −D(f).mean() | SN-GAN style |
| `wgangp` | D(f)−D(r) + λ·GP | −D(f) | Forces `norm=instance`, `n_critic=5`, betas 0.0/0.9 |

### WGAN-GP guardrails

- `build_discriminator` overrides `norm=batch` → `instance` with a warning; `BatchNorm2d` breaks the per-sample gradient penalty because statistics are computed across the interpolated batch.
- `train_gan.py::_apply_mode_defaults` sets `n_critic=5` and TTUR betas `(0.0, 0.9)` automatically when `--gan_mode wgangp` is passed and the user has not overridden those flags.

### Training step order

1. Generator forward (once).
2. Discriminator update × `n_critic`: real pair = `[cond, gt]`, fake pair = `[cond, pooled_detach(fake)]`, loss = 0.5·(real + fake); WGAN-GP adds gradient penalty.
3. Generator update: `lambda_recon · CombinedLoss + lambda_gan · adv [+ lambda_feat · feat_match]`. Adversarial term is skipped during `warmup_epochs`.

---

## Session 3 — 2026-06-19 — Shared evaluation metrics

### New file: `training/metrics.py`

Single public function `compute_metrics(pred, gt) -> {"raps": float, "ssim": float}`, decorated `@torch.no_grad()`, averaged over the batch. Tensors are `(B, T, H, W)`; both metrics sum over `T` before computing.

**RAPS.** Accumulates event frame (`sum(T)`), runs `rfft2`, bins `|F|²` into integer-radius rings using a device-cached radius map (centered frequencies: `fy ∈ [−H/2, H/2)`, `fx ∈ [0, W/2]`). Ring counts from `torch.bincount`. Returns `F.l1_loss(log(P_pred + ε), log(P_gt + ε))`.

**SSIM.** Sums over `T` → `(B, 1, H, W)` float32, Gaussian kernel (11×11, σ=1.5), `data_range` = per-sample GT max clamped to ε. Uses `pytorch_msssim.ssim` when importable; falls back to a self-contained implementation that shares the same formula. Both the kernel and the rfft2 radius map are cached per `(size, device)` / `(H, W, device)`.

### Wired into

| Location | Keys added |
|---|---|
| `Trainer._train_step` | `monitor/raps`, `monitor/ssim` (fed to `MetricTracker`, logged to wandb) |
| `GANTrainer._update_generator` | same |
| `validate()` | `Val/raps`, `Val/ssim` (averaged over full val set) |

### TP note

TP (`loss/tp`) is computed by `CombinedLoss` and logged through `MetricTracker` as before. It remains a logged monitor; the new RAPS/SSIM metrics are pure monitors that never enter any loss total.

### Verification smoke tests (run after install)

---

## Session 4 — 2026-06-29 — PixelTemporalCritic discriminator

### Motivation

The existing PatchGAN (`NLayerDiscriminator`) treats the 15-channel voxel grid as an opaque multi-channel image, collapsing the temporal dimension into channel space. It has no mechanism to reason about the temporal ordering or structure of event bins. `PixelTemporalCritic` preserves the time axis and lets an attention encoder compare temporal patterns per spatial super-pixel.

### Architecture

```
input (B, 30, H, W)  [cond ‖ target, concatenated by the training loop]
  │
  ├── spatial_stem  Conv3d(1, d, k=(1,s,s), stride=(1,s,s))   signed target enters directly, no pre-activation
  │     (B, 1, 15, H, W) → (B, d, 15, H', W')  — time axis stays length 15
  │     reshape → (B·H'·W', 15, d) temporal tokens per super-pixel
  │
  ├── pos_emb       nn.Parameter(15, d)          learned temporal positional embedding, init zeros
  │
  ├── cond_stem     Conv2d(C_cond, d, k=s, s=s)  lands on same (H', W') grid
  │     → (B·H'·W', 1, d)  broadcast + additive fusion across 15 temporal slots
  │
  ├── CLS token     nn.Parameter(1, 1, d)          prepended → sequence length 16
  │
  ├── L × TemporalBlock  pre-LN MHSA (manual matmul, no flash) + GELU FFN
  │     spectral_norm on all Q/K/V/out projections and FFN linears
  │
  └── head          Linear(d, 1) → reshape (B, 1, H', W')   raw logit, no sigmoid
```

At defaults (d=64, L=4, s=4, H=260, W=346): H'=65, W'=86, 5590 super-pixels, sequence length 16.

### Key constraints honoured

- **Signed inputs**: no rectification before `spatial_stem`; Conv3d receives raw ±-valued voxel bins
- **Time axis intact**: Conv3d kernel=(1,s,s) stride=(1,s,s) — temporal dim stays 15, only spatial is downsampled
- **Manual attention**: `q @ k.T → softmax → @ v` avoids flash-attention kernels that block `create_graph=True` (needed for WGAN-GP gradient penalty)
- **Spectral norm throughout**: stems, all attention projections (Q/K/V/out), all FFN linears, head

### `use_projection` flag

When `--use_projection` is set, an extra `Linear(d, d)` projects the conditioning vector and its inner product with the CLS output is added to the final logit. Adds a learnable bilinear interaction between the discriminator's summary and the conditioning context. Off by default.

### `build_discriminator` routing

`netD="temporal"` routes to `PixelTemporalCritic`. `ndf` maps to `d`; `--temporal_L` and `--temporal_s` control depth and stride. The pix2pix `init_weights` pass is skipped (it only covers `Conv2d`/`ConvTranspose2d`); the critic uses spectral norm + PyTorch defaults.

### New files / changed files

| File | Change |
|---|---|
| `model/discriminator.py` | Added `TemporalBlock`, `PixelTemporalCritic`; extended `build_discriminator` |
| `training/trainer_gan.py` | Added `temporal_L`, `temporal_s`, `use_projection` to `GANTrainerConfig`; passed to factory |
| `train_gan.py` | Added `--netD temporal`, `--temporal_L`, `--temporal_s`, `--use_projection` |

### Quick-start command

```bash
python train_gan.py --netD temporal --ndf 64 --temporal_L 4 --temporal_s 4 \
  --gan_mode lsgan --lambda_recon 100 --lambda_gan 1 \
  --warmup_epochs 3 --lr_g 2e-4 --lr_d 5e-5 \
  --validate --use_wandb
```

```python
# Forward pass through each netD variant on 30-ch dummy at 346×260
import torch
from model.discriminator import build_discriminator

for netD in ("basic", "pixel", "multiscale"):
    D = build_discriminator(netD=netD, in_channels_cond=15, in_channels_target=15).cuda()
    x = torch.randn(2, 30, 260, 346).cuda()
    out = D(x)
    print(netD, type(out).__name__)

# One full train step per gan_mode
from training.trainer_gan import GANTrainer, GANTrainerConfig
for mode in ("lsgan", "vanilla", "hinge", "wgangp"):
    cfg = GANTrainerConfig(gan_mode=mode, num_steps=1, batch_size=2, device="cuda")
    # … instantiate trainer and call _train_step with dummy tensors
```
