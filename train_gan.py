import argparse
import logging
from pathlib import Path

import wandb

from training.config_loader import FullConfig, TrainingConfig, load_config
from training.trainer import set_seed
from training.trainer_gan import GANTrainer

CONFIG_PATH_ARGS = ("model_config", "loss_config", "training_config")

# GAN knobs a CLI flag may override on top of the shared runtime set.
GAN_OVERRIDABLE_KEYS = frozenset(
    {"gan_mode", "lambda_gan", "lambda_recon", "lambda_gp", "lr_g", "lr_d", "n_critic"}
)

WGANGP_N_CRITIC = 5
WGANGP_BETAS    = (0.0, 0.9)
ADAM_BETAS      = (0.5, 0.999)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a voxel refinement backbone with a discriminator, from JSON configs"
    )

    configs = parser.add_argument_group("configs")
    configs.add_argument("--model_config",    type=str, required=True,
                         help="Path to a JSON model config, e.g. configs/models/swin_t7.json")
    configs.add_argument("--loss_config",     type=str, required=True,
                         help="Path to a JSON loss config, e.g. configs/losses/default.json")
    configs.add_argument("--training_config", type=str, required=True,
                         help="Path to a JSON GAN training config, e.g. configs/training/gan.json")

    # Runtime overrides. Defaults stay None so an unset flag never shadows the JSON.
    overrides = parser.add_argument_group("overrides")
    overrides.add_argument("--num_steps",      type=int)
    overrides.add_argument("--checkpoint_dir", type=str)
    overrides.add_argument("--seed",           type=int)
    overrides.add_argument("--validate",       action="store_true", default=None)
    overrides.add_argument("--use_wandb",      action="store_true", default=None)
    overrides.add_argument("--wandb_project",  type=str)
    overrides.add_argument("--wandb_name",     type=str)
    overrides.add_argument("--model_path",     type=str)
    overrides.add_argument("--continue_training", action="store_true", default=None)
    overrides.add_argument("--device",         type=str, choices=["cuda", "cpu"])
    overrides.add_argument("--batch_size",     type=int)

    gan = parser.add_argument_group("gan overrides")
    gan.add_argument("--gan_mode",     type=str,
                     choices=["vanilla", "lsgan", "hinge", "wgangp"])
    gan.add_argument("--lambda_gan",   type=float)
    gan.add_argument("--lambda_recon", type=float)
    gan.add_argument("--lambda_gp",    type=float)
    gan.add_argument("--n_critic",     type=int)
    gan.add_argument("--lr_g",         type=float)
    gan.add_argument("--lr_d",         type=float)

    return parser


def apply_mode_defaults(
    training:      TrainingConfig,
    cli_overrides: dict,
    logger:        logging.Logger | None = None,
) -> None:
    """Switch to wgangp-appropriate settings wherever the config left the generic ones.

    An explicit CLI value always wins; only untouched defaults move.
    """
    if training.gan_mode != "wgangp":
        return

    changes: dict[str, object] = {}
    if training.norm_D == "batch":
        changes["norm_D"] = "instance"
    if "n_critic" not in cli_overrides and training.n_critic == 1:
        changes["n_critic"] = WGANGP_N_CRITIC
    for net in ("g", "d"):
        betas = (training.get(f"beta1_{net}"), training.get(f"beta2_{net}"))
        if betas == ADAM_BETAS:
            changes[f"beta1_{net}"], changes[f"beta2_{net}"] = WGANGP_BETAS

    for key, value in changes.items():
        training.extra[key] = value
        if logger is not None:
            logger.info(f"wgangp default: {key} -> {value!r}")


def parse_config(argv: list[str] | None = None) -> tuple[FullConfig, str | None]:
    parser = build_parser()
    args   = parser.parse_args(argv)

    config_paths = {f"{name}_path": getattr(args, name) for name in CONFIG_PATH_ARGS}
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key not in CONFIG_PATH_ARGS and value is not None
    }

    logger = logging.getLogger(__name__)
    config = load_config(
        **config_paths,
        cli_overrides=overrides,
        logger=logger,
        extra_overridable=GAN_OVERRIDABLE_KEYS,
    )
    apply_mode_defaults(config.training, overrides, logger)
    return config, args.wandb_name


def init_wandb(config: FullConfig, run_name: str | None = None) -> None:
    if not config.training.use_wandb:
        return
    if run_name is None:
        run_name = Path(config.training.checkpoint_dir).name or "gan_training_run"
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
    trainer = GANTrainer(config.model, config.loss, config.training)
    final_checkpoint = trainer.train()
    print(f"Training finished. Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
