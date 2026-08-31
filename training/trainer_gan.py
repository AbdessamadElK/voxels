import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from data_loader.dsec_full import make_data_loader
from losses.gan_loss import GANLoss, gradient_penalty
from losses.v2ce_losses_v2 import CombinedLoss
from model.discriminator import build_discriminator
from model.unet_transformer import UNetTransformer
from utils import get_logger

from .metrics import compute_metrics
from .trainer import MetricTracker, resolve_device, set_seed  # noqa: F401 (re-exported)
from .validation import validate, visualize_output


# ---------------------------------------------------------------------------
# Defaults (shared with trainer.py)
# ---------------------------------------------------------------------------
DEFAULT_SUM_FREQ  = 100
DEFAULT_VIS_FREQ  = 1_000
DEFAULT_VAL_FREQ  = 5_000
DEFAULT_SAVE_FREQ = 10_000


# ---------------------------------------------------------------------------
# VoxelPool  (pix2pix ImagePool adapted for voxel grids)
# ---------------------------------------------------------------------------

class VoxelPool:
    """Ring buffer that probabilistically returns stored or current voxels.

    Size 0 disables buffering (always returns the current batch).
    """

    def __init__(self, pool_size: int = 50) -> None:
        self.pool_size = pool_size
        self.pool: list[torch.Tensor] = []

    def query(self, voxels: torch.Tensor) -> torch.Tensor:
        if self.pool_size == 0:
            return voxels
        out: list[torch.Tensor] = []
        for img in voxels.unbind(0):
            img = img.unsqueeze(0)
            if len(self.pool) < self.pool_size:
                self.pool.append(img.detach().clone())
                out.append(img)
            elif random.random() < 0.5:
                idx = random.randrange(len(self.pool))
                stored = self.pool[idx].clone()
                self.pool[idx] = img.detach().clone()
                out.append(stored)
            else:
                out.append(img)
        return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
# GANTrainerConfig
# ---------------------------------------------------------------------------

@dataclass
class GANTrainerConfig:
    # Training
    num_steps:      int   = 200_000
    checkpoint_dir: str   = "checkpoints_gan"
    validate:       bool  = False
    seed:           int   = 1
    device:         str   = "cuda"

    # Data
    batch_size:  int = 6
    num_workers: int = 8

    # Generator model
    in_channels: int       = 15
    dim:         int       = 24
    num_blocks:  list[int] = field(default_factory=lambda: [2, 3, 3, 4])

    # Discriminator
    netD:              str   = "basic"
    ndf:               int   = 64
    n_layers_D:        int   = 3
    norm_D:            str   = "instance"
    spectral_norm:     bool  = False
    num_D:             int   = 2
    conditional:       bool  = True
    return_interm_feats: bool = False
    init_type:         str   = "normal"
    init_gain:         float = 0.02
    temporal_L:        int   = 4
    temporal_s:        int   = 8
    use_projection:    bool  = False

    # GAN hyperparameters
    gan_mode:       str   = "lsgan"
    lambda_gan:     float = 1.0
    lambda_recon:   float = 10.0
    lambda_gp:      float = 10.0
    lambda_feat:    float = 0.0
    n_critic:       int   = 1
    warmup_epochs:  int   = 0
    pool_size:      int   = 50

    # Optimisers (TTUR)
    lr_g:     float = 1e-4
    lr_d:     float = 4e-4
    beta1_g:  float = 0.5
    beta2_g:  float = 0.999
    beta1_d:  float = 0.5
    beta2_d:  float = 0.999
    weight_decay: float = 1e-4

    # Reconstruction loss weights
    lambda_stp: float = 1.0
    lambda_tp:  float = 1.0
    lambda_ef:  float = 1.0
    lambda_ss:  float = 1.0
    lambda_ts:  float = 1.0

    # Logging / cadence
    use_wandb:     bool = False
    wandb_project: str  = "EV_SNN"
    sum_freq:  int = DEFAULT_SUM_FREQ
    vis_freq:  int = DEFAULT_VIS_FREQ
    val_freq:  int = DEFAULT_VAL_FREQ
    save_freq: int = DEFAULT_SAVE_FREQ

    # Checkpoint loading
    model_path:        str  = ""
    continue_training: bool = False

    @classmethod
    def from_args(cls, args) -> "GANTrainerConfig":
        return cls(**vars(args))


# ---------------------------------------------------------------------------
# GANTrainer
# ---------------------------------------------------------------------------

class GANTrainer:
    def __init__(self, config: GANTrainerConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)

        self.date_label = datetime.now().strftime("%Y-%m-%d")
        self.save_dir   = Path(config.checkpoint_dir) / self.date_label
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(str(self.save_dir / "train_gan.log"))
        self.logger.info("==== NEW GAN TRAINING PROCESS ====")
        self.logger.info(config)

        self.net_G        = self._build_generator()
        self.net_D        = self._build_discriminator()
        self.train_loader = self._build_train_loader()
        self.val_loader   = self._build_val_loader() if config.validate else None
        self.optim_G, self.optim_D = self._build_optimizers()
        self.criterion_recon = CombinedLoss(
            lambda_stp=config.lambda_stp,
            lambda_tp=config.lambda_tp,
            lambda_ef=config.lambda_ef,
            lambda_ss=config.lambda_ss,
            lambda_ts=config.lambda_ts,
        )
        self.criterion_gan = GANLoss(gan_mode=config.gan_mode)
        self.voxel_pool    = VoxelPool(pool_size=config.pool_size)
        self.scaler        = torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None
        self.metrics       = MetricTracker(config.use_wandb, config.sum_freq)

        self.start_step = self._maybe_load_checkpoint()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_generator(self) -> nn.Module:
        net = UNetTransformer(
            in_channels=self.config.in_channels,
            dim=self.config.dim,
            num_blocks=self.config.num_blocks,
        )
        self.logger.info(
            f"Generator: UNetTransformer(in_channels={self.config.in_channels}, "
            f"dim={self.config.dim}, num_blocks={self.config.num_blocks})"
        )
        return net.to(self.device)

    def _build_discriminator(self) -> nn.Module:
        net = build_discriminator(
            netD=self.config.netD,
            in_channels_cond=self.config.in_channels,
            in_channels_target=self.config.in_channels,
            conditional=self.config.conditional,
            ndf=self.config.ndf,
            n_layers=self.config.n_layers_D,
            num_D=self.config.num_D,
            norm=self.config.norm_D,
            use_spectral_norm=self.config.spectral_norm,
            return_interm_feats=self.config.return_interm_feats,
            init_type=self.config.init_type,
            init_gain=self.config.init_gain,
            gan_mode=self.config.gan_mode,
            temporal_L=self.config.temporal_L,
            temporal_s=self.config.temporal_s,
            use_projection=self.config.use_projection,
        )
        self.logger.info(
            f"Discriminator: netD={self.config.netD}, ndf={self.config.ndf}, "
            f"n_layers={self.config.n_layers_D}, norm={self.config.norm_D}"
        )
        return net.to(self.device)

    def _build_train_loader(self):
        phase  = "train" if self.config.validate else "trainval"
        loader = make_data_loader(
            phase, batch_size=self.config.batch_size, num_workers=self.config.num_workers
        )
        self.logger.info("Train loader created.")
        return loader

    def _build_val_loader(self):
        loader = make_data_loader("val", batch_size=1, num_workers=0)
        self.logger.info("Validation loader created.")
        return loader

    def _build_optimizers(self) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        optim_G = torch.optim.Adam(
            self.net_G.parameters(),
            lr=self.config.lr_g,
            betas=(self.config.beta1_g, self.config.beta2_g),
            weight_decay=self.config.weight_decay,
        )
        optim_D = torch.optim.Adam(
            self.net_D.parameters(),
            lr=self.config.lr_d,
            betas=(self.config.beta1_d, self.config.beta2_d),
            weight_decay=self.config.weight_decay,
        )
        return optim_G, optim_D

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> str:
        self.net_G.train()
        self.net_D.train()
        step          = self.start_step
        keep_training = True
        epoch         = 0

        while keep_training:
            progress = tqdm(self.train_loader, total=len(self.train_loader), ncols=80)
            for voxel, voxel_gt, _ in progress:
                losses = self._train_step(voxel, voxel_gt, epoch)
                progress.set_description(
                    f"Step {step}/{self.config.num_steps} | "
                    f"G={losses.get('G/total', 0):.3f} "
                    f"D_real={losses.get('D/real', 0):.3f}"
                )
                self.metrics.update(losses, step=step)
                step += 1

                if self._should_visualize(step):
                    self._log_visualizations(step)
                if self._should_validate(step):
                    self._run_validation(step)
                if self._should_save(step):
                    self._save_training_checkpoint(step)
                if step >= self.config.num_steps:
                    keep_training = False
                    break

            epoch += 1
            time.sleep(0.03)

        self.metrics.flush(step)
        return self._save_final_model()

    def _train_step(
        self,
        voxel:    torch.Tensor,
        voxel_gt: torch.Tensor,
        epoch:    int,
    ) -> dict[str, float]:
        voxel    = voxel.to(self.device, non_blocking=True).float()
        voxel_gt = voxel_gt.to(self.device, non_blocking=True).float()

        # ----------------------------------------------------------
        # Forward through generator (once; reused for both updates)
        # ----------------------------------------------------------
        if self.scaler is not None:
            with torch.amp.autocast("cuda"):
                fake = self.net_G(voxel)                           # (B, C, H, W)
        else:
            fake = self.net_G(voxel)

        out: dict[str, float] = {}

        # ----------------------------------------------------------
        # Discriminator update  (n_critic times)
        # ----------------------------------------------------------
        for _ in range(self.config.n_critic):
            pooled_fake = self.voxel_pool.query(fake.detach())
            d_losses = self._update_discriminator(voxel, voxel_gt, pooled_fake)
        out.update(d_losses)

        # ----------------------------------------------------------
        # Generator update
        # ----------------------------------------------------------
        g_losses = self._update_generator(voxel, voxel_gt, fake, epoch)
        out.update(g_losses)

        return out

    def _discriminator_input(
        self, cond: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.config.conditional:
            return torch.cat([cond, target], dim=1)   # (B, 2C, H, W)
        return target

    def _update_discriminator(
        self,
        cond:      torch.Tensor,
        real:      torch.Tensor,
        fake_pool: torch.Tensor,
    ) -> dict[str, float]:
        self.optim_D.zero_grad(set_to_none=True)

        real_inp = self._discriminator_input(cond, real)
        fake_inp = self._discriminator_input(cond, fake_pool)

        d_real = self.net_D(real_inp)
        d_fake = self.net_D(fake_inp)

        loss_real = self.criterion_gan(d_real, target_is_real=True,  for_discriminator=True)
        loss_fake = self.criterion_gan(d_fake, target_is_real=False, for_discriminator=True)
        loss_D    = 0.5 * (loss_real + loss_fake)

        gp_val = real.new_zeros(1).squeeze()
        if self.config.gan_mode == "wgangp":
            gp_val = gradient_penalty(
                netD=self.net_D,
                real_target=real,
                fake_target=fake_pool,
                cond=cond if self.config.conditional else None,
                lambda_gp=self.config.lambda_gp,
                conditional=self.config.conditional,
            )
            loss_D = loss_D + gp_val

        loss_D.backward()
        torch.nn.utils.clip_grad_norm_(self.net_D.parameters(), max_norm=1.0)
        self.optim_D.step()

        # scalar monitor values from D outputs
        def _mean_score(d_out):
            if isinstance(d_out, list):
                scores = [o[0] if isinstance(o, tuple) else o for o in d_out]
                return float(torch.stack([s.mean() for s in scores]).mean())
            if isinstance(d_out, tuple):
                return float(d_out[0].mean())
            return float(d_out.mean())

        return {
            "D/real":     _mean_score(d_real),
            "D/fake":     _mean_score(d_fake),
            "D/loss":     float(loss_D),
            "D/gp":       float(gp_val),
        }

    def _update_generator(
        self,
        cond:    torch.Tensor,
        real:    torch.Tensor,
        fake:    torch.Tensor,
        epoch:   int,
    ) -> dict[str, float]:
        self.optim_G.zero_grad(set_to_none=True)

        in_warmup = epoch < self.config.warmup_epochs

        # Reconstruction loss (spectral terms stay in float32 inside the modules)
        if self.scaler is not None:
            with torch.amp.autocast("cuda"):
                recon_losses = self.criterion_recon(fake, real)
        else:
            recon_losses = self.criterion_recon(fake, real)

        loss_G = self.config.lambda_recon * recon_losses["total"]

        g_adv_val = fake.new_zeros(1).squeeze()
        feat_val  = fake.new_zeros(1).squeeze()

        if not in_warmup:
            fake_inp = self._discriminator_input(cond, fake)
            d_fake_for_G = self.net_D(fake_inp)
            g_adv_val = self.criterion_gan(d_fake_for_G, target_is_real=True, for_discriminator=False)
            loss_G = loss_G + self.config.lambda_gan * g_adv_val

            # Feature matching (only when return_interm_feats and lambda_feat > 0)
            if self.config.return_interm_feats and self.config.lambda_feat > 0.0:
                real_inp = self._discriminator_input(cond, real)
                with torch.no_grad():
                    d_real_feats = self.net_D(real_inp)

                def _extract_feats(d_out):
                    if isinstance(d_out, list):
                        all_feats = []
                        for o in d_out:
                            if isinstance(o, tuple):
                                all_feats.extend(o[1])
                        return all_feats
                    if isinstance(d_out, tuple):
                        return d_out[1]
                    return []

                real_feats = _extract_feats(d_real_feats)
                fake_feats = _extract_feats(d_fake_for_G)
                for rf, ff in zip(real_feats, fake_feats):
                    feat_val = feat_val + torch.nn.functional.l1_loss(ff, rf.detach())
                loss_G = loss_G + self.config.lambda_feat * feat_val

        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(self.net_G.parameters(), max_norm=1.0)
        self.optim_G.step()

        m = compute_metrics(fake.detach(), real)
        return {
            "G/adv":         float(g_adv_val),
            "G/feat":        float(feat_val),
            "G/recon":       float(recon_losses["total"]),
            "G/total":       float(loss_G),
            "loss/stp":      float(recon_losses["stp"]),
            "loss/tp":       float(recon_losses["tp"]),
            "loss/ef":       float(recon_losses["ef"]),
            "loss/ss":       float(recon_losses["ss"]),
            "loss/ts":       float(recon_losses["ts"]),
            "monitor/raps":  m["raps"],
            "monitor/ssim":  m["ssim"],
        }

    # ------------------------------------------------------------------
    # Periodic hooks
    # ------------------------------------------------------------------

    def _should_visualize(self, step: int) -> bool:
        return self.config.use_wandb and step > 0 and step % self.config.vis_freq == 0

    def _should_validate(self, step: int) -> bool:
        return (
            self.config.validate
            and self.val_loader is not None
            and step > 0
            and step % self.config.val_freq == 0
        )

    def _should_save(self, step: int) -> bool:
        return step > 0 and step % self.config.save_freq == 0

    @torch.no_grad()
    def _log_visualizations(self, step: int) -> None:
        self.net_G.eval()
        for index, frame in visualize_output(self.net_G, self.device):
            wandb.log({f"progress_{index + 1}": wandb.Image(frame)}, step=step)
        self.net_G.train()

    @torch.no_grad()
    def _run_validation(self, step: int) -> None:
        self.net_G.eval()
        val_metrics = validate(self.net_G, self.val_loader, self.device)
        if self.config.use_wandb:
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)
        self.net_G.train()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _checkpoint_state(self, step: int) -> dict[str, Any]:
        return {
            "step":                    step,
            "model_state_dict":        self.net_G.state_dict(),
            "discriminator_state_dict": self.net_D.state_dict(),
            "optimizer_G_state_dict":  self.optim_G.state_dict(),
            "optimizer_D_state_dict":  self.optim_D.state_dict(),
            "metrics":                 self.metrics.state_dict(),
            "config":                  self.config,
        }

    def _save_training_checkpoint(self, step: int) -> str:
        path = self.save_dir / f"checkpoint_{step}.pth"
        torch.save(self._checkpoint_state(step), path)
        self.logger.info(f"Saved training checkpoint -> '{path}'.")
        return str(path)

    def _save_final_model(self) -> str:
        path = self.save_dir / "checkpoint.pth"
        torch.save(self._checkpoint_state(self.config.num_steps), path)
        self.logger.info(f"Saved final checkpoint -> '{path}'.")
        return str(path)

    def _maybe_load_checkpoint(self) -> int:
        if not self.config.model_path:
            if self.config.continue_training:
                self.logger.warning(
                    "continue_training=True but no model_path provided. Starting from scratch."
                )
            return 0

        path = Path(self.config.model_path)
        if not path.is_file():
            self.logger.warning(f"No checkpoint at '{path}'. Starting from scratch.")
            return 0

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self._load_model_weights(checkpoint)

        start_step = 0
        if self.config.continue_training:
            start_step = self._load_training_state(checkpoint)

        self.logger.info(f"Loaded checkpoint from '{path}'.")
        return start_step

    def _load_model_weights(self, checkpoint: dict[str, Any]) -> None:
        g_state = checkpoint.get("model_state_dict", checkpoint)
        self.net_G.load_state_dict(g_state, strict=False)
        if "discriminator_state_dict" in checkpoint:
            self.net_D.load_state_dict(checkpoint["discriminator_state_dict"], strict=False)

    def _load_training_state(self, checkpoint: dict[str, Any]) -> int:
        required = {"optimizer_G_state_dict", "optimizer_D_state_dict", "step"}
        missing  = required.difference(checkpoint.keys())
        if missing:
            self.logger.warning(
                f"Cannot resume training — checkpoint missing: {sorted(missing)}. "
                "Loaded weights only."
            )
            return 0

        self.optim_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        self.optim_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        if "metrics" in checkpoint:
            self.metrics.load_state_dict(checkpoint["metrics"])
        return int(checkpoint["step"])
