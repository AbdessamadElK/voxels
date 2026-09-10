import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from data_loader.dsec_full import make_data_loader
from losses.loss_registry import LossRegistry
from model.backbone_registry import BackboneRegistry
from utils import get_logger

from .config_loader import LossConfig, ModelConfig, TrainingConfig
from .metrics import compute_metrics
from .validation import validate, visualize_output


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_SUM_FREQ  = 100
DEFAULT_VIS_FREQ  = 1_000
DEFAULT_VAL_FREQ  = 5_000
DEFAULT_SAVE_FREQ = 10_000

MAX_GRAD_NORM   = 1.0
ADAM_BETAS      = (0.9, 0.999)
SCHEDULER_SLACK = 100
WARMUP_FRACTION = 0.01


# -----------------------------------------------------------------------------
# Legacy configuration
# -----------------------------------------------------------------------------
@dataclass
class TrainerConfig:
    """Pre-refactor config, kept so checkpoints that pickled it still unpickle.

    New runs use ModelConfig / LossConfig / TrainingConfig from config_loader.
    """

    num_steps:      int   = 200_000
    checkpoint_dir: str   = "checkpoints"
    lr:             float = 2e-4
    weight_decay:   float = 1e-4
    validate:       bool  = False
    seed:           int   = 1
    device:         str   = "cuda"

    batch_size:  int = 6
    num_workers: int = 8

    in_channels: int       = 15
    dim:         int       = 24
    num_blocks:  list[int] = field(default_factory=lambda: [2, 3, 3, 4])

    lambda_stp: float = 1.0
    lambda_tp:  float = 1.0
    lambda_ef:  float = 1.0
    lambda_ss:  float = 1.0
    lambda_ts:  float = 1.0

    use_wandb:     bool = False
    wandb_project: str  = "EV_SNN"
    sum_freq:  int = DEFAULT_SUM_FREQ
    vis_freq:  int = DEFAULT_VIS_FREQ
    val_freq:  int = DEFAULT_VAL_FREQ
    save_freq: int = DEFAULT_SAVE_FREQ

    model_path:        str  = ""
    continue_training: bool = False


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


class MetricTracker:
    """Accumulates metrics and periodically logs averaged values."""

    def __init__(self, use_wandb: bool, log_frequency: int) -> None:
        self.use_wandb     = use_wandb
        self.log_frequency = log_frequency
        self.running: dict[str, float] = {}
        self.count = 0

    def update(self, metrics: dict[str, float], step: int) -> None:
        if not metrics:
            return
        self.count += 1
        for k, v in metrics.items():
            self.running[k] = self.running.get(k, 0.0) + float(v)
        if step > 0 and step % self.log_frequency == 0:
            self.flush(step)

    def flush(self, step: int) -> None:
        if self.count == 0:
            return
        averaged = {k: v / self.count for k, v in self.running.items()}
        if self.use_wandb:
            wandb.log(averaged, step=step)
        self.running.clear()
        self.count = 0

    def state_dict(self) -> dict[str, Any]:
        return {"running": self.running, "count": self.count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.running = state.get("running", {})
        self.count   = state.get("count", 0)


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
class Trainer:
    def __init__(
        self,
        model_config:    ModelConfig,
        loss_config:     LossConfig,
        training_config: TrainingConfig,
    ) -> None:
        self.model_config    = model_config
        self.loss_config     = loss_config
        self.training_config = training_config
        self.device = resolve_device(training_config.device)

        self.date_label = datetime.now().strftime("%Y-%m-%d")
        self.save_dir   = Path(training_config.checkpoint_dir) / self.date_label
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(str(self.save_dir / "train.log"))
        self.logger.info("==== NEW TRAINING PROCESS ====")
        self.logger.info(f"Model config:    {model_config.as_dict()}")
        self.logger.info(f"Loss config:     {loss_config.as_dict()}")
        self.logger.info(f"Training config: {training_config.as_dict()}")

        self.model        = self._build_model()
        self.train_loader = self._build_train_loader()
        self.val_loader   = self._build_val_loader() if training_config.validate else None
        self.optimizer    = self._build_optimizer()
        self.scheduler    = self._build_scheduler()
        self.criterion    = LossRegistry.build(loss_config.as_dict(), logger=self.logger)
        self.scaler  = torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None
        self.metrics = MetricTracker(training_config.use_wandb, training_config.sum_freq)

        self.start_step = self._maybe_load_checkpoint()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _build_model(self) -> nn.Module:
        model = BackboneRegistry.build(self.model_config.as_dict(), device=self.device)
        self.logger.info(
            f"Backbone '{self.model_config.backbone}' "
            f"({sum(p.numel() for p in model.parameters()):,} parameters)"
        )
        return model

    def _build_train_loader(self):
        phase  = "train" if self.training_config.validate else "trainval"
        loader = make_data_loader(
            phase,
            batch_size=self.training_config.batch_size,
            num_workers=self.training_config.num_workers,
        )
        self.logger.info("Train loader created.")
        return loader

    def _build_val_loader(self):
        loader = make_data_loader("val", batch_size=1, num_workers=0)
        self.logger.info("Validation loader created.")
        return loader

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.lr,
            betas=ADAM_BETAS,
            weight_decay=self.training_config.weight_decay,
        )

    def _build_scheduler(self):
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.training_config.lr,
            total_steps=self.training_config.num_steps + SCHEDULER_SLACK,
            pct_start=WARMUP_FRACTION,
            cycle_momentum=False,
            anneal_strategy="linear",
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self) -> str:
        self.model.train()
        step         = self.start_step
        keep_training = True

        while keep_training:
            progress = tqdm(self.train_loader, total=len(self.train_loader), ncols=80)
            for voxel, voxel_gt, _ in progress:
                losses = self._train_step(voxel, voxel_gt)
                progress.set_description(
                    f"Step {step}/{self.training_config.num_steps} | "
                    f"loss={losses['loss/total']:.4f}"
                )
                self.metrics.update(losses, step=step)
                step += 1

                if self._should_visualize(step):
                    self._log_visualizations(step)
                if self._should_validate(step):
                    self._run_validation(step)
                if self._should_save(step):
                    self._save_training_checkpoint(step)
                if step >= self.training_config.num_steps:
                    keep_training = False
                    break

            time.sleep(0.03)

        self.metrics.flush(step)
        return self._save_final_model()

    def _train_step(
        self,
        voxel:    torch.Tensor,
        voxel_gt: torch.Tensor,
    ) -> dict[str, float]:
        voxel    = voxel.to(self.device, non_blocking=True).float()
        voxel_gt = voxel_gt.to(self.device, non_blocking=True).float()

        self.optimizer.zero_grad(set_to_none=True)

        if self.scaler is not None:
            with torch.amp.autocast("cuda"):
                pred   = self.model(voxel)
                losses = self.criterion(pred, voxel_gt)
            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=MAX_GRAD_NORM)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            pred   = self.model(voxel)
            losses = self.criterion(pred, voxel_gt)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=MAX_GRAD_NORM)
            self.optimizer.step()

        self.scheduler.step()

        m = compute_metrics(pred.detach(), voxel_gt)
        return {
            "loss/total":    float(losses["total"]),
            "loss/stp":      float(losses["stp"]),
            "loss/tp":       float(losses["tp"]),
            "loss/ef":       float(losses["ef"]),
            "loss/ss":       float(losses["ss"]),
            "loss/ts":       float(losses["ts"]),
            "monitor/raps":  m["raps"],
            "monitor/ssim":  m["ssim"],
        }

    # ------------------------------------------------------------------
    # Periodic hooks
    # ------------------------------------------------------------------
    def _should_visualize(self, step: int) -> bool:
        return (
            self.training_config.use_wandb
            and step > 0
            and step % self.training_config.vis_freq == 0
        )

    def _should_validate(self, step: int) -> bool:
        return (
            self.training_config.validate
            and self.val_loader is not None
            and step > 0
            and step % self.training_config.val_freq == 0
        )

    def _should_save(self, step: int) -> bool:
        return step > 0 and step % self.training_config.save_freq == 0

    @torch.no_grad()
    def _log_visualizations(self, step: int) -> None:
        self.model.eval()
        for index, frame in visualize_output(self.model, self.device):
            wandb.log({f"progress_{index + 1}": wandb.Image(frame)}, step=step)
        self.model.train()

    @torch.no_grad()
    def _run_validation(self, step: int) -> None:
        self.model.eval()
        val_metrics = validate(self.model, self.val_loader, self.device)
        if self.training_config.use_wandb:
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)
        self.model.train()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _checkpoint_state(self, step: int) -> dict[str, Any]:
        return {
            "step":                 step,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler":            self.scheduler.state_dict(),
            "metrics":              self.metrics.state_dict(),
            "config": {
                "model":    self.model_config.as_dict(),
                "loss":     self.loss_config.as_dict(),
                "training": self.training_config.as_dict(),
            },
        }

    def _save_training_checkpoint(self, step: int) -> str:
        path = self.save_dir / f"checkpoint_{step}.pth"
        torch.save(self._checkpoint_state(step), path)
        self.logger.info(f"Saved training checkpoint -> '{path}'.")
        return str(path)

    def _save_final_model(self) -> str:
        path = self.save_dir / "checkpoint.pth"
        torch.save(self._checkpoint_state(self.training_config.num_steps), path)
        self.logger.info(f"Saved final checkpoint -> '{path}'.")
        return str(path)

    def _maybe_load_checkpoint(self) -> int:
        if not self.training_config.model_path:
            if self.training_config.continue_training:
                self.logger.warning(
                    "continue_training=True but no model_path provided. Starting from scratch."
                )
            return 0

        path = Path(self.training_config.model_path)
        if not path.is_file():
            self.logger.warning(f"No checkpoint at '{path}'. Starting from scratch.")
            return 0

        # weights_only=False: pre-refactor checkpoints pickled a TrainerConfig instance.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self._load_model_weights(checkpoint)

        start_step = 0
        if self.training_config.continue_training:
            start_step = self._load_training_state(checkpoint)

        self.logger.info(f"Loaded checkpoint from '{path}'.")
        return start_step

    def _load_model_weights(self, checkpoint: dict[str, Any]) -> None:
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state, strict=False)

    def _load_training_state(self, checkpoint: dict[str, Any]) -> int:
        required = {"optimizer_state_dict", "scheduler", "step"}
        missing  = required.difference(checkpoint.keys())
        if missing:
            self.logger.warning(
                f"Cannot resume training — checkpoint missing: {sorted(missing)}. "
                "Loaded weights only."
            )
            return 0

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        if "metrics" in checkpoint:
            self.metrics.load_state_dict(checkpoint["metrics"])
        return int(checkpoint["step"])
