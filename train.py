import argparse
import logging
from pathlib import Path

import wandb

from training.config_loader import FullConfig, load_config
from training.trainer import Trainer, set_seed

CONFIG_PATH_ARGS = ("model_config", "loss_config", "training_config")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a voxel refinement backbone from JSON configs"
    )

    configs = parser.add_argument_group("configs")
    configs.add_argument("--model_config",    type=str, required=True,
                         help="Path to a JSON model config, e.g. configs/models/swin_t7.json")
    configs.add_argument("--loss_config",     type=str, required=True,
                         help="Path to a JSON loss config, e.g. configs/losses/default.json")
    configs.add_argument("--training_config", type=str, required=True,
                         help="Path to a JSON training config, e.g. configs/training/default.json")

    # Runtime overrides. Defaults stay None so an unset flag never shadows the JSON.
    overrides = parser.add_argument_group("overrides")
    overrides.add_argument("--num_steps",      type=int)
    overrides.add_argument("--checkpoint_dir", type=str)
    overrides.add_argument("--lr",             type=float)
    overrides.add_argument("--seed",           type=int)
    overrides.add_argument("--validate",       action="store_true", default=None)
    overrides.add_argument("--use_wandb",      action="store_true", default=None)
    overrides.add_argument("--wandb_project",  type=str)
    overrides.add_argument("--wandb_name",     type=str)
    overrides.add_argument("--model_path",     type=str)
    overrides.add_argument("--continue_training", action="store_true", default=None)
    overrides.add_argument("--device",         type=str, choices=["cuda", "cpu"])
    overrides.add_argument("--batch_size",     type=int)

    return parser


def parse_config(argv: list[str] | None = None) -> tuple[FullConfig, str | None]:
    parser = build_parser()
    args   = parser.parse_args(argv)

    config_paths = {f"{name}_path": getattr(args, name) for name in CONFIG_PATH_ARGS}
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key not in CONFIG_PATH_ARGS and value is not None
    }

    config = load_config(
        **config_paths,
        cli_overrides=overrides,
        logger=logging.getLogger(__name__),
    )
    return config, args.wandb_name


def init_wandb(config: FullConfig, run_name: str | None = None) -> None:
    if not config.training.use_wandb:
        return
    if run_name is None:
        run_name = Path(config.training.checkpoint_dir).name or "training_run"
    wandb.init(
        project=config.training.wandb_project,
        name=run_name,
        config=config.as_dict(),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config, wandb_name = parse_config()
    set_seed(config.training.seed)
    init_wandb(config, wandb_name)
    trainer = Trainer(config.model, config.loss, config.training)
    final_checkpoint = trainer.train()
    print(f"Training finished. Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
