import argparse
from pathlib import Path

import wandb

from training.trainer import (
    Trainer,
    TrainerConfig,
    set_seed,
    DEFAULT_SUM_FREQ,
    DEFAULT_VIS_FREQ,
    DEFAULT_VAL_FREQ,
    DEFAULT_SAVE_FREQ,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train UNetTransformer for synthetic-to-real voxel refinement")

    # Training
    training = parser.add_argument_group("training")
    training.add_argument("--num_steps",      type=int,   default=200_000)
    training.add_argument("--checkpoint_dir", type=str,   default="checkpoints")
    training.add_argument("--lr",             type=float, default=2e-4)
    training.add_argument("--weight_decay",   type=float, default=1e-4)
    training.add_argument("--validate",       action="store_true")
    training.add_argument("--seed",           type=int,   default=1)
    training.add_argument("--device",         type=str,   default="cuda", choices=["cuda", "cpu"])

    # Data
    data = parser.add_argument_group("data")
    data.add_argument("--batch_size",  type=int, default=6)
    data.add_argument("--num_workers", type=int, default=8)

    # Model
    model = parser.add_argument_group("model")
    model.add_argument("--time_bins",   type=int,   default=15)
    model.add_argument("--dim",         type=int,   default=48)
    model.add_argument("--num_blocks",  type=int,   nargs="+", default=[4, 6, 6, 8],
                       help="Block counts per level: L1 L2 bottleneck [unused]")
    model.add_argument("--lambda_stp",  type=float, default=1.0)
    model.add_argument("--lambda_tp",   type=float, default=1.0)
    model.add_argument("--lambda_ef",   type=float, default=1.0)

    # Logging
    logging = parser.add_argument_group("logging")
    logging.add_argument("--use_wandb",     action="store_true")
    logging.add_argument("--wandb_project", type=str, default="EV_SNN")
    logging.add_argument("--wandb_name",    type=str, default=None)
    logging.add_argument("--sum_freq",      type=int, default=DEFAULT_SUM_FREQ)
    logging.add_argument("--vis_freq",      type=int, default=DEFAULT_VIS_FREQ)
    logging.add_argument("--val_freq",      type=int, default=DEFAULT_VAL_FREQ)
    logging.add_argument("--save_freq",     type=int, default=DEFAULT_SAVE_FREQ)

    # Checkpoint
    checkpoint = parser.add_argument_group("checkpoint")
    checkpoint.add_argument("--model_path",        type=str,  default="")
    checkpoint.add_argument("--continue_training", action="store_true")

    return parser


def parse_config() -> tuple[TrainerConfig, str | None]:
    parser = build_parser()
    args   = parser.parse_args()
    wandb_name = args.wandb_name
    delattr(args, "wandb_name")
    config = TrainerConfig.from_args(args)
    return config, wandb_name


def init_wandb(config: TrainerConfig, run_name: str | None = None) -> None:
    if not config.use_wandb:
        return
    if run_name is None:
        run_name = Path(config.checkpoint_dir).name or "training_run"
    wandb.init(project=config.wandb_project, name=run_name, config=config.__dict__)


def main() -> None:
    config, wandb_name = parse_config()
    set_seed(config.seed)
    init_wandb(config, wandb_name)
    trainer = Trainer(config)
    final_checkpoint = trainer.train()
    print(f"Training finished. Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
