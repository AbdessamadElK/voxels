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

---

## Session 5 — 2026-09-07 — Configuration-driven architecture

Training is now driven by JSON config files instead of long argparse invocations.
CLI flags survive only as overrides for runtime parameters.

### New files

| File | Purpose |
| --- | --- |
| `model/swin_transformer.py` | Swin Transformer T/7 backbone, symmetric encoder-decoder |
| `model/backbone_registry.py` | `BackboneRegistry` — name -> model class, builds from a config dict |
| `losses/loss_registry.py` | `LossRegistry` — same pattern for losses |
| `training/config_loader.py` | JSON loading, dataclass configs, CLI merge |
| `configs/models/*.json` | Per-backbone architecture |
| `configs/losses/default.json` | Loss selection and weights |
| `configs/training/{default,gan}.json` | Runtime and optimiser settings |

### Config layering

Three sources, lowest to highest priority:

1. Hardcoded dataclass defaults in `config_loader.py`
2. The JSON files
3. CLI flags — **only** for keys in `OVERRIDABLE_KEYS`

`OVERRIDABLE_KEYS` covers `num_steps`, `checkpoint_dir`, `lr`, `seed`, `validate`,
`use_wandb`, `wandb_project`, `wandb_name`, `model_path`, `continue_training`,
`device`, `batch_size`. `train_gan.py` adds `gan_mode`, `lambda_gan`, `lambda_recon`,
`lambda_gp`, `lr_g`, `lr_d`, `n_critic` via `load_config(extra_overridable=...)`.

Architecture and loss weights are JSON-only. Passing `--dim` or `--lambda_ef` logs a
warning and changes nothing. Every load and every override is logged at INFO.

Keys a JSON file carries beyond the declared dataclass fields land in `config.extra`
and read back as plain attributes, so `training_config.netD` works without declaring
a GAN-specific dataclass.

### Available backbones

`BackboneRegistry.list_available()` -> `['swin_t7', 'unet_transformer']`

`build()` filters the config dict to the constructor's accepted arguments and raises
on an unknown backbone, an unsupported key, or an `out_channels` the backbone cannot
honour (`UNetTransformer` ties its output width to `in_channels`).

### Swin T/7 geometry

Encoder: patch embed (4×4) then three `PatchMerging` steps — spatial dims drop **32×**
overall, channels `96 -> 192 -> 384 -> 768`. Decoder mirrors it with `PatchExpand`,
concatenating the encoder output at each level and fusing through a linear layer.
Depths `[2, 2, 6, 2]` / heads `[3, 6, 12, 24]`; the decoder reverses both, so its
depths are `[2, 6, 2, 2]` and heads `[24, 12, 6, 3]` (head dim stays 32 throughout).
55.6M parameters.

Arbitrary input sizes work: `forward` pads H and W up to a multiple of 32 and crops
the output back. Inside each block, feature maps pad up to a multiple of the window
size, and the attention mask marks both the shifted-window regions and the padded
keys, so real tokens never attend to padding.

### patch_embed carries no bias

`PatchEmbed`'s convolution is `bias=False`. A bias immediately before the LayerNorm is
nearly cancelled by the normalisation, yet it accumulates the gradient summed over
every token: measured at **2.59e6** on a real batch, 250× the next-largest gradient in
the network, which overflowed fp16 under autocast and drove the GAN generator to NaN
on the first step. Dropping it puts the maximum gradient at 8.2e3. The LayerNorm's own
bias supplies the shift.

### Backward compatibility

`TrainerConfig` and `GANTrainerConfig` remain in `trainer.py` / `trainer_gan.py`
solely so pre-refactor checkpoints, which pickled those dataclasses, still unpickle —
hence `weights_only=False` on load. New checkpoints store the config as three plain
dicts under `"config"` (`model`, `loss`, `training`).

Old `UNetTransformer` weights load into a registry-built model with no missing or
unexpected keys.

### Quick-start commands

```bash
# UNetTransformer, unchanged behaviour
python train.py \
  --model_config configs/models/unet_transformer.json \
  --loss_config configs/losses/default.json \
  --training_config configs/training/default.json

# Swin T/7, short run to a scratch directory
python train.py \
  --model_config configs/models/swin_t7.json \
  --loss_config configs/losses/default.json \
  --training_config configs/training/default.json \
  --num_steps 5000 --checkpoint_dir ckpts/swin --validate --use_wandb

# GAN, Swin generator
python train_gan.py \
  --model_config configs/models/swin_t7.json \
  --loss_config configs/losses/default.json \
  --training_config configs/training/gan.json \
  --lambda_recon 50 --checkpoint_dir checkpoints_gan/swin

# Resume
python train.py ... --model_path ckpts/swin/2026-09-07/checkpoint_10000.pth --continue_training
```

### Verified

- Swin forward at 256×256, 260×346 (DSEC), 128×192; encoder resolutions `[(64,64),(32,32),(16,16),(8,8)]`
- Drop path stochastic in `train()`, deterministic in `eval()`; shifts alternate `[0,3,0,3,0,3]`
- Zeroing a skip fusion's encoder half changes the output — skips are live
- All 343 parameters receive gradients; AMP forward/backward clean at 260×346
- `train.py` runs end to end on real DSEC data with both backbones
- `train_gan.py` + Swin: 300 steps, no NaN, `G/total` 2.6e3 -> ~1.1e3
- Resume restores step, weights, optimiser and scheduler state
- Old GAN checkpoint (`checkpoints_gan/2026-06-29`) still unpickles and loads

### Known hazard — GAN trainer has no GradScaler

`GANTrainer._train_step` runs the generator forward under `autocast("cuda")` but calls
`loss_G.backward()` / `loss_D.backward()` directly, with no `GradScaler`. `Trainer`
(non-GAN) does scale. Without scaling there is no inf detection and no skip-step, so a
single overflowing gradient passes through `clip_grad_norm_` — an infinite total norm
makes `clip_coef` zero, `inf * 0` is NaN, and Adam writes NaN into the weights.

`UNetTransformer` survives this only because it uses `bias=False` throughout and its
gradients run ~30× smaller. Any future backbone with pre-norm biases is exposed.
Left unchanged here (refactor scope excluded GAN training logic); adding a `GradScaler`
to `trainer_gan.py` mirroring `trainer.py` is the fix if it recurs.

---

## Session 6 — 2026-09-08 — Swin encoder + UNet decoder

### Change

Replaced the symmetric Swin decoder (window attention + `PatchExpand`) with a
convolutional UNet decoder. Attention stays in the encoder; decoding is bilinear
upsampling, skip concatenation and convolutions.

Note on the premise: the symmetric decoder was measured working in Session 5 (200
steps, loss 18.96 -> 14.04, RAPS 3.45 -> 1.24, gradients reaching all 343 params).
This change was requested on architectural grounds, not to repair a broken forward
pass. It is a defensible simplification — fewer parameters, a decoder that is easy to
reason about — but it was not a bug fix.

### Architecture

```
Input (B, 15, H, W)
  -> PatchEmbed 4x4            (B,  96, H/4,  W/4)
  -> SwinEncoder stages        skips: 96 @ H/4, 192 @ H/8, 384 @ H/16, 768 @ H/32
  -> SwinUNetDecoder
       up + cat skip[2] -> fuse 1152->384 -> ConvBlock   (B, 384, H/16, W/16)
       up + cat skip[1] -> fuse  576->192 -> ConvBlock   (B, 192, H/8,  W/8)
       up + cat skip[0] -> fuse  288-> 96 -> ConvBlock   (B,  96, H/4,  W/4)
       up x4 -> ConvBlock -> output_proj                 (B,  15, H,    W)
```

This decoder was band-limited to H/4 and produced blurred output; Session 7 adds the
full-resolution stem and global residual that fix it. Current diagram:

```
```

33.19M parameters: 27.54M encoder, 5.66M decoder (was 55.64M symmetric).

### Files

| File | Change |
| --- | --- |
| `model/swin_transformer.py` | `SwinEncoder` (returns bottleneck + skips) and a thin `SwinTransformer` wrapper; deleted `PatchExpand`, `FinalPatchExpand`, decoder stages |
| `model/unet_decoder.py` | [NEW] `SwinUNetDecoder` |
| `model/backbone_registry.py` | imports `SwinEncoder`; `**kwargs` backbones now receive all config keys |
| `model/__init__.py` | exports `SwinEncoder`, `SwinUNetDecoder` |
| `configs/models/swin_t7.json` | dropped `patch_size`, added `attn_drop` / `qkv_bias` |

### Two places the spec did not match the code

1. **`ConvBlock` cannot change channel count.** The spec called
   `ConvBlock(in_ch, out_ch, bias=False)`, but the real signature is
   `ConvBlock(dim, bias=False)` — a channel-preserving residual block. Each decoder
   level therefore uses a 3x3 `fuse` conv for the reduction, then `ConvBlock` at the
   target width. `ConvBlock` itself is untouched.

2. **The spec's decoder stopped at H/4.** Its shape annotations ended at
   `(B, out_channels, H/4, W/4)` while claiming `(B, 15, H, W)`. Added a x4 upsample of
   the *features* before the output projection, so the 15-channel output is not
   upsampled directly.

Upsampling uses `F.interpolate(size=skip.shape[2:])` rather than `scale_factor=2`,
matching `UNetTransformer` — identical for even dims, exact for odd ones.

### output_proj initialisation

`output_proj` is initialised `trunc_normal_(std=0.01)`, not Kaiming. Kaiming's
fan_out/ReLU gain is meant for hidden layers; on the final projection it started the
output far from the target, and `lambda_recon=100` amplified the gradient to the edge
of fp16:

| output_proj init | recon at step 0 | max abs grad | fp16 headroom | GAN step 0 |
| --- | --- | --- | --- | --- |
| Kaiming (fan_out) | 177 | 6.28e4 | 1.0x | overflow -> all 188 tensors NaN |
| x0.1 | 41.2 | 8.0e3 | 8.2x | clean |
| trunc_normal 0.01 | 31.1 | 1.6e3 | 41x | clean |

It also improved plain training over 200 steps: loss 314.6 -> 26.8 became
37.5 -> 17.0, SSIM 0.035 -> 0.313.

### Verified

- `SwinEncoder` alone: skips `[(2,96,64,64), (2,192,32,32), (2,384,16,16), (2,768,8,8)]`, bottleneck `(2,768,8,8)` == `skips[3]`, all 168 params get grads
- `SwinUNetDecoder` alone: bottleneck + skips -> `(2,15,256,256)`; fuse convs `1152->384, 576->192, 288->96`; all 20 params get grads
- Full model: 256x256, 260x346 (DSEC), 128x192 all round-trip; asymmetric `out_channels=3` works
- Drop path stochastic in train, deterministic in eval; strict `state_dict` round-trip over 188 tensors
- Zeroing the skip half of `fuse[0]` changes the output — skips are live
- `train.py` on real DSEC: 200 steps, loss 37.5 -> 17.0, RAPS 3.17 -> 1.61, SSIM 0.156 -> 0.313, all 188 params updated, 11/200 steps skipped by GradScaler
- Resume restores step, all 188 weights, optimiser and scheduler
- `train_gan.py`: 100 steps clean, no NaN
- `unet_transformer` unchanged and still builds, runs and loads old checkpoints

### GAN trainer still has no GradScaler

The Session 5 hazard stands and this session hit it: `GANTrainer` runs the generator
forward under autocast but calls `backward()` unscaled, so one overflowing gradient
gives `clip_grad_norm_` an infinite total norm, `clip_coef` becomes 0, `inf * 0` is
NaN, and Adam writes NaN into every parameter. The Kaiming output head above triggered
exactly that on step 0.

The small init restores ~41x headroom, but it does not remove the hazard — gradient
spikes above 1e5 were still observed mid-run (in fp32 parameters, where they are
harmless). Adding a `GradScaler` to `trainer_gan.py`, mirroring `trainer.py`, remains
the real fix.

---

## Session 7 — 2026-09-08 — GAN GradScaler + full-resolution decoder path

Two independent fixes: the GAN trainer's missing loss scaling, and blurred Swin output.

### 1. GradScaler in the GAN trainer

`GANTrainer._update_generator` now scales the generator loss before backward, mirroring
`Trainer`:

```python
self.scaler.scale(loss_G).backward()
self.scaler.unscale_(self.optim_G)
torch.nn.utils.clip_grad_norm_(self.net_G.parameters(), max_norm=MAX_GRAD_NORM)
self.scaler.step(self.optim_G)
self.scaler.update()
```

Only the generator update is scaled. The discriminator's forward runs outside autocast
on fp32 inputs (`torch.cat` promotes the fp16 generator output back to fp32), so its
backward carries no fp16 overflow risk and needs no scaler.

Verified by reinstating the Kaiming output head that used to NaN every parameter on
step 0: with the scaler it survives 40 steps, the scaler skipping the overflowing ones.

**Watch the skip rate.** With `lambda_recon=100` the generator loss sits near 3e3 and
the scaler settles at scale 1.0, skipping ~16/40 steps — safe, but ~40% of updates are
discarded. Lowering `lambda_recon` would recover them.

### 2. Swin output was band-limited to H/4

`patch_embed` downsamples 4x immediately, so the finest encoder feature map is H/4 and
the decoder's only route back to full resolution was a bilinear upsample. The model
could not represent per-pixel structure, and event voxel grids are almost entirely
per-pixel high frequency.

Overfitting a single batch (400 steps, plain L1) isolates representational capacity
from optimisation:

| | L1 | RAPS | SSIM | out absmax |
| --- | --- | --- | --- | --- |
| input vs gt (baseline) | 0.2515 | 1.44 | 0.386 | 9.40 |
| swin, before fix | 0.0832 | **5.29** | 0.800 | 2.16 |
| gt downsampled 4x + upsampled | 0.0702 | **4.13** | 0.903 | — |
| swin, after fix | 0.0943 | **1.96** | 0.812 | 4.22 |

The pre-fix RAPS (5.29) matched the 4x-blur ceiling (4.13), not the baseline (1.44):
the output was spectrally a blurred image. Low L1 was misleading — event voxels are
mostly zero, so a smooth low-amplitude field (absmax 2.16 against a target of 16.6)
scores well on L1 while looking nothing like event data.

**Fix — two full-resolution paths in `SwinUNetDecoder`:**

- `stem`: a 3x3 convolution on the input at full resolution, concatenated after the x4
  upsample, so per-pixel detail reaches the output without passing through the encoder.
- global residual: `output_proj` predicts a correction added to the input when
  `in_channels == out_channels`. With the small output init the model starts as an
  identity, which is the right prior for refinement.

The full-resolution head runs at `embed_dim // 2` channels, not `embed_dim`, to keep
activations at H x W affordable.

`SwinUNetDecoder.forward` now takes `(bottleneck, skips, inp)` and crops the encoder
padding itself, so `SwinTransformer.forward` no longer crops.

### Results

Overfit RAPS 5.29 -> 1.96, well below the 4.13 band-limit ceiling. Over 300 real
training steps the model now starts near identity (step-0 RAPS 1.49, was 3.17) and in
eval beats the input baseline on SSIM (0.411 vs 0.315) with RAPS on par (0.841 vs
0.819).

33.24M parameters (27.54M encoder, 5.71M decoder), 190 tensors.

### Verified

- Zeroing `output_proj` makes the model an exact identity — the residual path is wired
- Zeroing `stem` changes the output — the full-resolution path is live
- Shapes hold at 256x256, 260x346, 128x192 and 100x130 (non-multiple of 32)
- `out_channels != in_channels` disables the residual and still runs
- Strict `state_dict` round-trip over 190 tensors; drop path train-only
- GAN: 40 steps clean on both the normal and the deliberately broken init

---

## Session 8 — 2026-09-08 — Configurable decoder depth

`SwinUNetDecoder` had exactly one `ConvBlock` per level and one at full resolution,
thin next to `UNetTransformer`'s `[2, 3, 3, 4]`. Added `decoder_blocks` (per level,
deepest first) and `final_blocks` (full-resolution count), both defaulting to 1 so the
default model is unchanged.

Encoder depth needed no code — `depths`, `num_heads` and `embed_dim` already reach
`SwinEncoder` through the config.

### Cost of depth (batch 4, 260x346, AMP)

| config | params | encoder | decoder | s/step | VRAM |
| --- | --- | --- | --- | --- | --- |
| default | 33.24M | 27.54M | 5.71M | 0.074 | 1.72 GB |
| decoder [2,2,2] + 2 final | 33.65M | 27.54M | 6.11M | 0.072 | 1.93 GB |
| decoder [4,4,4] + 3 final | 34.45M | 27.54M | 6.91M | 0.076 | 2.18 GB |
| Swin-S depths [2,2,18,2] | 54.56M | 48.85M | 5.71M | 0.123 | 2.39 GB |
| Swin-S + decoder [2,2,2]/2 | 54.96M | 48.85M | 6.11M | 0.113 | 2.59 GB |
| Swin-B dim 128 [2,2,18,2] | 96.90M | 86.77M | 10.13M | 0.110 | 3.45 GB |

Two findings worth keeping:

- **Decoder depth is nearly free.** `[4,4,4]` plus 3 final blocks costs 1.2M parameters
  and 3% step time — ConvBlock is depthwise-separable and every level below the last
  runs at reduced resolution.
- **Swin-B is faster than Swin-S** despite 78% more parameters (0.110 vs 0.123 s/step).
  Channel widths of 128/256/512/1024 hit tensor cores better than 96/192/384/768.
  Widening beats deepening on this hardware.

### Depth is probably not the current bottleneck

At step 2500 the model beats input passthrough on SSIM (0.545 vs 0.317, 146/150
validation samples) but loses on RAPS (1.069 vs 0.747, 10/150) and under-predicts
amplitude badly (peak 6.6 against a ground truth of 21.2).

Systematic under-amplitude on sparse data is the signature of an L1-dominated
objective — the conditional median of a mostly-zero target is near zero — not of
insufficient capacity. A deeper model under the same loss will under-shoot more
precisely. Re-balancing `lambda_ss` / `lambda_ts` against `lambda_stp` / `lambda_ef`
via the calibration recipe above is the first thing to try.

The exception is `final_blocks`: the full-resolution head is where high-frequency
detail is synthesised, and one ConvBlock there is genuinely thin.

### state_dict note

`refine` became an `nn.Sequential`, so its keys moved from `refine.block.*` to
`refine.0.block.*`. Swin checkpoints written before this session will not load; the
`unet_transformer` backbone is unaffected.

---

## Session 9 — 2026-09-08 — TemporalSpectralLoss epsilon

### The historical value: ts_eps = 1e-8

**Every run up to and including 2026-09-08 used `eps = 1e-8` in `TemporalSpectralLoss`.**
To reproduce any of them, train with `configs/losses/legacy_ts_eps.json`, which pins
that value. It is also the constructor default of `CombinedLoss` and the module-level
constant `LEGACY_TS_EPS` in `losses/v2ce_losses_v2.py`, so nothing changes silently for
code that builds the loss directly. Only the config default moved.

### Why it changed

`TemporalSpectralLoss` computes `log(|F| + eps)` per pixel, where `F` is the temporal
FFT. Measured on validation data:

- ~45% of GT temporal spectrum bins are **exactly zero** (on a single sparse batch,
  81% fall below 1e-8); only 1.7% of bins on event-carrying pixels are zero
- informative magnitudes sit at p10 = 0.51, median 1.38, p90 = 4.94

`d/dx log(x + eps) = 1/(x + eps)`, so at `eps = 1e-8` the derivative reaches 1e8 exactly
where the target is empty. A network output is dense and small, the target is exactly
zero, and the term drives those predictions down with enormous force. **99.9% of the ts
gradient mass landed on pixels with no GT events at all.**

Consequences, all measured:

| | ts gradient vs other four terms |
| --- | --- |
| at initialisation | 16x the mean (65x `stp`) |
| at step 2500 | 149x `stp` |

The imbalance *grows* during training — as the prediction shrinks, `1/(x+eps)` grows,
which shrinks it further. A self-reinforcing collapse, and the reason the model peaked
at amplitude 6.6 against a GT of 21.2 and lost to input passthrough on RAPS.

Scaling the trained prediction by alpha confirmed the objective was pulling the wrong
way: total loss preferred alpha = 0.5, `ts` preferred 0.5, while RAPS preferred 1.5 and
`ss`/`stp` preferred 1.5/1.25 — the two terms wanting correct amplitude were the ones
being drowned out.

### Choosing 1e-3

| eps | ts value | ts \|grad\| | ratio to other terms' mean |
| --- | --- | --- | --- |
| 1e-8 (legacy) | 10.12 | 0.1334 | 16.02 |
| 1e-4 | 5.96 | 0.0237 | 2.85 |
| **1e-3** | **4.92** | **0.0077** | **0.93** |
| 1e-2 | 3.87 | 0.0025 | 0.30 |

`1e-3` brings ts into line with the other four terms (0.93x) while sitting ~500x below
the informative spectrum floor (p10 = 0.51), so real signal is untouched and only the
empty bins are floored.

`SpatialSpectralLoss` needs no such change: it works on the frame accumulated over T,
which is rarely exactly zero, and its gradient is flat at 0.030 across every eps from
1e-8 to 1e-1. It was left alone and has no eps parameter.

### Configs

| file | lambda_ts | ts_eps | purpose |
| --- | --- | --- | --- |
| `losses/default.json` | 1.0 | 1e-3 | fixed objective |
| `losses/legacy_ts_eps.json` | 1.0 | 1e-8 | reproduce pre-2026-09-08 runs |
| `losses/ts_balanced.json` | 0.01 | 1e-8 | down-weighting instead of fixing eps |

`lambda_ts` returns to 1.0 in the default config: with the epsilon corrected the term
no longer needs suppressing. Down-weighting could never have been more than a patch —
the gradient ratio drifts 65x -> 149x during training, so no fixed lambda holds it.

### Experiment: three loss arms, 2500 steps each, seed 1, batch 4

All arms trained from scratch under identical code. 150 validation samples,
GT peak amplitude 21.21.

| arm | RAPS ↓ | SSIM ↑ | L1 ↓ | peak | RAPS win | SSIM win |
| --- | --- | --- | --- | --- | --- | --- |
| input passthrough | **0.7467** | 0.3167 | 0.4216 | 10.12 | — | — |
| A legacy `ts_eps=1e-8, l_ts=1.0` | 1.0424 | 0.5430 | 0.3656 | 6.59 | 10/150 | 146/150 |
| B weight `ts_eps=1e-8, l_ts=0.01` | 1.0550 | 0.5644 | 0.4368 | **10.34** | 10/150 | 147/150 |
| C eps fix `ts_eps=1e-3, l_ts=1.0` | 1.1302 | **0.5838** | **0.3574** | 6.29 | 8/150 | 147/150 |

**Confirmed:** `ts` caused the amplitude collapse. Down-weighting it (B) lifted peak
amplitude 6.59 -> 10.34.

**Not confirmed:** that fixing `ts` would fix RAPS. No arm beats input passthrough on
RAPS, and arm C — the principled epsilon fix — came out slightly worse than both. The
prediction that C would preserve useful supervision that B discards did not hold.

Caveat: single runs, no seed variance. The RAPS spread across arms (1.04–1.13) is
small enough to be noise; the amplitude gap (6.3 vs 10.3) is not.

### The real constraint: correction magnitude

The model has a global residual, so `out = inp + correction`. Measured:

```
||correction|| / ||input||  =  0.99 (arm B),  0.97 (arm C)
```

The correction is as large as the input — the model replaces the signal rather than
refining it. Blending it back (`inp + alpha * correction`) is strictly monotonic in
both metrics:

| alpha | RAPS (B) | SSIM (B) |
| --- | --- | --- |
| 0.00 (pure input) | 0.7137 | 0.3358 |
| 0.50 | 0.8393 | 0.4327 |
| 1.00 (as trained) | 0.9983 | 0.5713 |

There is no sweet spot: every step toward the model's output improves SSIM and degrades
RAPS. The two metrics are in direct conflict for this model, which produces a smooth,
structurally-aligned field while RAPS rewards the spiky sparse statistics the input
already has.

No re-weighting among the five current terms addresses this — all five are satisfied by
a smooth prediction. Closing the RAPS gap needs either supervision that explicitly
rewards sparsity/spikiness (value-distribution or histogram matching, an L0-style
term), or more high-frequency capacity, or an explicit decision that SSIM matters more
than RAPS for the downstream task.

### Keep the epsilon fix regardless

`1/(x + eps) = 1e8` on ~45% of bins is a defect independent of what it does to RAPS: it
made one term 149x the others by step 2500 and drove the amplitude collapse. `1e-3`
keeps the term's informative content at full weight. Arm C's slightly worse RAPS is
within run-to-run noise and does not argue for reinstating 1e-8.

---

## Session 10 — 2026-09-09 — RAPSLoss, and a second validation scheme

### Why re-weighting kept failing

`_raps` in `training/metrics.py` is differentiable (`scatter_add_`), so each loss term's
gradient can be compared directly against the RAPS gradient. Measured on validation
data at the trained checkpoint:

| term | cos(grad, grad_RAPS) | effective pull = cos x \|grad\| |
| --- | --- | --- |
| stp | -0.0060 | -0.00024 |
| tp | -0.0008 | -0.00002 |
| ef | **-0.1012** | -0.00110 |
| ss | **+0.0575** | +0.00184 |
| ts | -0.0057 | -0.00022 |

Every cosine is near zero: the objective is close to orthogonal to RAPS, and `ef`
opposes it. Net pull across all five terms, +0.0003. Rescaling near-orthogonal vectors
cannot move the model along RAPS, which is why the `lambda_ts` and `ts_eps` arms shifted
amplitude but left RAPS flat.

### New file: `losses/raps_loss.py`

`RAPSLoss` — L1 on the log radially-averaged power spectrum of the accumulated event
frame. `SpatialSpectralLoss` was the nearest existing term but over-constrains: it pins
every frequency bin, while RAPS constrains only the radial profile.

Wired into `CombinedLoss` as `lambda_raps`, default `0.0`, and the FFT is skipped when
the weight is zero, so nothing changes for existing configs.

Verified: matches `training.metrics._raps` to 2.4e-7 and its gradient has cosine
1.000000 with the metric's. Unlike `ts`, it is eps-safe — ring means sum many bins, so
the minimum observed ring power is 4761 with a median of 1.3e6, and the gradient is
identical across eps from 1e-8 to 1e-2.

Optimising RAPS makes it a training objective, not an independent monitor. SSIM and the
visual comparison carry the honest check now.

### lambda_raps sweep, 2500 steps, 150 validation samples

| arm | RAPS | SSIM | peak | RAPS win | SSIM win |
| --- | --- | --- | --- | --- | --- |
| input passthrough | 0.7467 | 0.3167 | 10.12 | — | — |
| 0 | 1.1302 | 0.5838 | 6.29 | 8/150 | 147/150 |
| 0.5 | 0.7755 | 0.5609 | 6.74 | 51/150 | 146/150 |
| **2.0** | **0.6317** | 0.5445 | 7.50 | **132/150** | **147/150** |
| 8.0 | 0.5632 | 0.4974 | 7.75 | 139/150 | 145/150 |

`lambda_raps = 2.0` is the chosen default: 44% better RAPS than `0` for 6.7% less SSIM,
and the first setting to beat input passthrough on both metrics at once. `8.0` buys 11%
more RAPS for 8.6% more SSIM, a worse trade.

Full validation set (248 samples) at `lambda_raps = 2.0`:

```
                  RAPS      SSIM        L1
  input         0.7528    0.3135    0.4169
  output        0.6355    0.5402    0.3625
  improved     214/248   240/248   212/248
```

### New file: `evaluate.py`

Scores a checkpoint against the input it refines, both against ground truth, so the
question "does the network beat doing nothing" has a direct answer. Reads the model
config from the checkpoint, loads strictly, defaults to `weights_only=True`, and builds
its own loader with `shuffle=False, drop_last=False` — `make_data_loader` shuffles and
drops the last partial batch, which suits training and not scoring.

### Occupancy needs care

The first version reported occupancy as `|x| > 1e-6`, which was misleading. Input and
ground truth hold real zeros (83% and 76% on the full set), so their occupancy is
threshold-insensitive. A model output holds **none**, so its figure is set entirely by
the cutoff: 1.000 above 1e-6, 0.935 above 1e-2, 0.489 above 0.1.

`evaluate.py` now reports `exact 0`, which needs no threshold, beside occupancy at a
`--occupancy_threshold` the caller can change (default 0.1, under the median GT event
magnitude of 0.64 and above numerical dust).

The corrected reading of the density problem: the model **spreads energy** — over-dense
at low magnitudes, under-dense at high ones, at the same time.

| threshold | input | output | gt |
| --- | --- | --- | --- |
| >1e-2 | 0.189 | 0.935 | 0.288 |
| >0.1 | 0.182 | 0.489 | 0.272 |
| >0.5 | 0.142 | **0.093** | 0.181 |

Median nonzero magnitude: output 0.090 against ground truth 0.64.

### Still open

At `lambda_raps = 2.0` the peak is 7.45 against a ground truth of 21.13, occupancy
above 0.1 is 0.482 against 0.229, and the output still contains no exact zeros. The
spectral profile matches; the amplitude distribution does not. A sparsity or
value-distribution term is the next lever, not more capacity.


---

## Session 11 — 2026-09-10 — Factorized space-time UNet

A second backbone, opt-in and selected by config. Nothing on the existing path changed
behaviour. It is the first model here that treats the 15 time bins as a **volume axis**
rather than as channels.

### New files

| file | holds |
| --- | --- |
| `model/factorized_st_unet.py` | `FactorizedSTUNet` plus `AxisAttention`, `FactorizedConvBlock`, `FactorizedAttentionBlock` |
| `configs/models/factorized_st.json` | divided attention arm |
| `configs/models/factorized_st_axial.json` | axial attention arm |
| `configs/baseline_unet.sh`, `configs/factorized_st_divided.sh`, `configs/factorized_st_axial.sh` | launch scripts, one checkpoint dir and wandb name each |
| `smoke_factorized_st.py` | shape and gradient checks for all three backbones, no dataset needed |

### The design

The loader hands over `(B, 15, H, W)` where the 15 channels are the time bins, and
every existing loss and metric reads dim 1 as T. The model keeps that contract at both
ends and works on a volume in between: a stem lifts `(B, 1, T, H, W)` to
`(B, dim, T, H, W)`, and `output_proj` projects back. **T is never downsampled** — only
H and W, by 2, twice. `forward` takes and returns one tensor, so `validation.py`,
`visualize_output` and `evaluate.py` need no change.

**Factorized convolutions.** Every block is R(2+1)D: a spatial `(1, 3, 3)` conv, GELU,
then a temporal `(3, 1, 1)` conv, with a residual. A joint `(3, 3, 3)` kernel costs 3x
the parameters for the same receptive field and puts no nonlinearity between the two
directions.

**Factorized attention.** `AxisAttention` folds every axis but one into the batch, so
an axis of length L costs L² attention rather than (T·H·W)². Two arrangements:

- `divided` — time, then the whole frame. TimeSformer's divided space-time.
- `axial` — time, then H, then W.

Both run at the bottleneck only (H/4 = 65 x 87), where `decoder_attn` can add one more
block at H/2.

Global spatial attention over a `(B·T)` batch is what forces the issue. At the
bottleneck the frame is 5655 tokens and the batch is B·T = 90, so a materialized
attention matrix is 23 GB. `F.scaled_dot_product_attention` never materializes it, and
that alone is what makes the `divided` arm runnable — the `nn.MultiheadAttention` used
by `TransformerBlock` would OOM. `axial` never faces the question: its longest axis is
87.

### The occupancy path, built and dropped the same day

The first version of this session shipped a sparsity mechanism with it: per-frame
occupancy logits predicted at each decoder level, gating that level's encoder skip
before the concat, with the output as `sigmoid(occ_0) ⊙ recon` so the model could reach
exact zeros. `losses/occupancy_loss.py` supervised every level (deep supervision) with
a choice of BCE or Tversky against the max-pooled nonzero mask of the ground truth.

It was cut before any experiment ran. **The density problem from Session 10 goes to a
GAN instead**, so a second, hand-designed sparsity prior in the generator would have
been a confound rather than a help. What is left is the architecture: R(2+1)D convs,
factorized bottleneck attention, plain concat skips.

Recorded here because the decision is the useful part. If it comes back, the shape it
had: gate `skip * (floor + (1 - floor) * sigmoid(occ_l))` with `occ_gate_floor` 0.1,
targets by spatial max-pool (not average — a coarse cell asks whether **any** event
falls in it), levels averaged so `lambda_occ` survives a change in decoder depth, and
Tversky `alpha` 0.7 / `beta` 0.3 to charge a false positive more than a miss.

One finding from building it outlived it, and turned out to matter for the model that
shipped. See below.

### Single-channel heads and fan_out

`kaiming_normal_(mode="fan_out")` counts the **output** channels, so on a head that
emits one channel it hands back a large standard deviation. The occupancy heads showed
it first — BCE 15.06 at init, sigmoid pinned on its flat tails with no gradient to
follow — and `output_proj` has the same shape, `(width2 -> 1)`.

Measured against a real sample, output standard deviation as a multiple of the ground
truth's:

| | kaiming fan_out | xavier |
| --- | --- | --- |
| `factorized_st` divided | 88.2x | 11.5x |
| `factorized_st` axial | 61.6x | 18.2x |
| `unet_transformer` (baseline, unchanged) | 16.5x | — |

`output_proj` now takes a xavier init back after the model-wide kaiming sweep. Without
it the model starts 5x further from the target than the baseline does and spends its
first steps shrinking the output — an init handicap that would have been read as an
architecture result. The baseline's own `output_proj` is `(dim*2 -> 15)`, where fan_out
counts 15 channels and the same rule is far kinder; it was left alone.

### Config

`FactorizedSTUNet` registers as backbone `factorized_st`, so the registry builds it from
JSON with no wiring. It pairs with the existing `CombinedLoss` like every other backbone.

| key | meaning | default |
| --- | --- | --- |
| `dim` | width at full resolution; levels widen 1x / 2x / 4x | 16 |
| `num_blocks` | conv blocks at encoder level 1, encoder level 2, bottleneck | `[1, 2, 2]` |
| `num_heads` | heads per level; index 2 drives the bottleneck | `[1, 2, 4]` |
| `attn_type` | `divided` or `axial` | `divided` |
| `decoder_attn` | one attention block at decoder level 2 (H/2) | `false` |

### Cost (260x346, AMP, RTX 3090)

| model | batch | params | s/step | peak VRAM |
| --- | --- | --- | --- | --- |
| unet_transformer (baseline) | 6 | 0.473M | 0.305 | 19.45 GB |
| factorized_st divided | 1 | 0.300M | 0.204 | 2.45 GB |
| factorized_st divided | 2 | 0.300M | 0.397 | 4.86 GB |
| factorized_st divided | 4 | 0.300M | 0.797 | 9.69 GB |
| factorized_st divided | 6 | 0.300M | 1.188 | 14.52 GB |
| factorized_st axial | 2 | 0.333M | 0.362 | 5.16 GB |
| factorized_st axial | 4 | 0.333M | 0.525 | 10.28 GB |
| factorized_st axial | 6 | 0.333M | 0.784 | 15.40 GB |

Batch 6 fits in 5 GB **less** than the baseline, so the launch scripts leave the batch
alone and the arms stay step-for-step comparable. The cost is time: 3.9x the baseline
per step at batch 6. `axial` is a third faster than `divided` at batch 4 and above (the
gap closes at batch 2, where neither saturates the card) for 11% more parameters —
worth reading as its own arm, not just a fallback.

Fewer parameters than the baseline (0.300M against 0.473M) at 3.9x the step time is the
shape of the trade: the volume is 15x the activations, and the work went into the
temporal axis rather than into width.

### Verification

```
# Backward compatibility: existing configs, 3 real steps
python train.py --model_config configs/models/unet_transformer.json \
  --loss_config configs/losses/default.json \
  --training_config configs/training/default.json --num_steps 3
# -> UNetTransformer (473,400 parameters), checkpoint written

# New backbone, 3 real steps under AMP
python train.py --model_config configs/models/factorized_st.json \
  --loss_config configs/losses/default.json \
  --training_config configs/training/default.json --num_steps 3

# Shapes and gradients for all three backbones, no dataset needed
python smoke_factorized_st.py
```

`validate` and `visualize_output` were run against the new backbone unchanged, and both
attention types were run under `autocast`, with and without `decoder_attn`.

One thing the smoke test deliberately does not assert: gradients are finite in fp32,
but at init the **scaled** fp16 gradient overflows and `GradScaler` skips the first few
steps while it halves the scale. The baseline does the same — its gradient norm at init
is 900, needing a scale below 73, against 65536 to start — so this is the loss stack,
not the new model.

### Next

- **GAN generator.** `FactorizedSTUNet` is a drop-in generator for `trainer_gan.py`,
  which this session left alone, and that is where the density problem is headed.
  `PixelTemporalCritic` was built to judge exactly the amplitude distribution Session 10
  found wrong.
- **`decoder_attn` is untested as a training arm.** It forwards and backwards correctly
  but nothing has run with it; attention at H/2 is 22490 tokens per frame under
  `divided`, so try it with `axial` first.
