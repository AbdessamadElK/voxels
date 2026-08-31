import argparse
from pathlib import Path

import wandb

from training.trainer import set_seed
from training.trainer_gan import (
    GANTrainer,
    GANTrainerConfig,
    DEFAULT_SUM_FREQ,
    DEFAULT_VIS_FREQ,
    DEFAULT_VAL_FREQ,
    DEFAULT_SAVE_FREQ,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train UNetTransformer + PatchGAN discriminator (pix2pix-style)"
    )

    # Training
    training = parser.add_argument_group("training")
    training.add_argument("--num_steps",      type=int,   default=200_000)
    training.add_argument("--checkpoint_dir", type=str,   default="checkpoints_gan")
    training.add_argument("--validate",       action="store_true")
    training.add_argument("--seed",           type=int,   default=1)
    training.add_argument("--device",         type=str,   default="cuda", choices=["cuda", "cpu"])

    # Data
    data = parser.add_argument_group("data")
    data.add_argument("--batch_size",  type=int, default=6)
    data.add_argument("--num_workers", type=int, default=8)

    # Generator model
    model = parser.add_argument_group("model")
    model.add_argument("--in_channels", type=int,   default=15)
    model.add_argument("--dim",         type=int,   default=24)
    model.add_argument("--num_blocks",  type=int,   nargs="+", default=[2, 3, 3, 4],
                       help="Block counts per level: L1 L2 bottleneck [unused]")

    # Discriminator
    disc = parser.add_argument_group("discriminator")
    disc.add_argument("--netD",         type=str,  default="basic",
                      choices=["basic", "pixel", "multiscale", "temporal"])
    disc.add_argument("--ndf",          type=int,  default=64)
    disc.add_argument("--n_layers_D",   type=int,  default=3)
    disc.add_argument("--norm_D",       type=str,  default="instance",
                      choices=["batch", "instance", "none"])
    disc.add_argument("--spectral_norm", action="store_true")
    disc.add_argument("--num_D",        type=int,  default=2)
    disc.add_argument("--conditional",    dest="conditional", action="store_true",  default=True)
    disc.add_argument("--no-conditional", dest="conditional", action="store_false")
    disc.add_argument("--return_interm_feats", action="store_true")
    disc.add_argument("--temporal_L",   type=int,  default=4,
                      help="PixelTemporalCritic encoder depth")
    disc.add_argument("--temporal_s",   type=int,  default=4,
                      help="PixelTemporalCritic spatial stride (super-pixel size)")
    disc.add_argument("--use_projection", action="store_true",
                      help="Add CLS·conditioning inner-product term to the logit")
    disc.add_argument("--init_type",    type=str,  default="normal",
                      choices=["normal", "xavier", "kaiming", "orthogonal"])
    disc.add_argument("--init_gain",    type=float, default=0.02)

    # GAN hyperparameters
    gan = parser.add_argument_group("gan")
    gan.add_argument("--gan_mode",       type=str,  default="lsgan",
                     choices=["vanilla", "lsgan", "hinge", "wgangp"])
    gan.add_argument("--lambda_gan",     type=float, default=1.0)
    gan.add_argument("--lambda_recon",   type=float, default=100.0)
    gan.add_argument("--lambda_gp",      type=float, default=10.0)
    gan.add_argument("--lambda_feat",    type=float, default=0.0)
    gan.add_argument("--n_critic",       type=int,   default=1)
    gan.add_argument("--warmup_epochs",  type=int,   default=0)
    gan.add_argument("--pool_size",      type=int,   default=50)

    # Reconstruction loss weights
    recon = parser.add_argument_group("reconstruction_losses")
    recon.add_argument("--lambda_stp", type=float, default=1.0)
    recon.add_argument("--lambda_tp",  type=float, default=1.0)
    recon.add_argument("--lambda_ef",  type=float, default=1.0)
    recon.add_argument("--lambda_ss",  type=float, default=1.0)
    recon.add_argument("--lambda_ts",  type=float, default=1.0)

    # Optimisers (TTUR)
    optim = parser.add_argument_group("optimizers")
    optim.add_argument("--lr_g",     type=float, default=1e-4)
    optim.add_argument("--lr_d",     type=float, default=4e-4)
    optim.add_argument("--beta1_g",  type=float, default=0.5)
    optim.add_argument("--beta2_g",  type=float, default=0.999)
    optim.add_argument("--beta1_d",  type=float, default=0.5)
    optim.add_argument("--beta2_d",  type=float, default=0.999)
    optim.add_argument("--weight_decay", type=float, default=1e-4)

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


def _apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Force wgangp-specific defaults when --gan_mode wgangp is selected."""
    if args.gan_mode == "wgangp":
        if args.norm_D == "batch":
            args.norm_D = "instance"
        # Only override n_critic if the user left it at the global default
        if args.n_critic == 1:
            args.n_critic = 5
        # TTUR betas for wgangp: 0.0 / 0.9
        if args.beta1_g == 0.5 and args.beta2_g == 0.999:
            args.beta1_g = 0.0
            args.beta2_g = 0.9
        if args.beta1_d == 0.5 and args.beta2_d == 0.999:
            args.beta1_d = 0.0
            args.beta2_d = 0.9
    return args


def parse_config() -> tuple[GANTrainerConfig, str | None]:
    parser    = build_parser()
    args      = parser.parse_args()
    args      = _apply_mode_defaults(args)
    wandb_name = args.wandb_name
    delattr(args, "wandb_name")
    config = GANTrainerConfig.from_args(args)
    return config, wandb_name


def init_wandb(config: GANTrainerConfig, run_name: str | None = None) -> None:
    if not config.use_wandb:
        return
    if run_name is None:
        run_name = Path(config.checkpoint_dir).name or "gan_training_run"
    wandb.init(project=config.wandb_project, name=run_name, config=config.__dict__)


def main() -> None:
    config, wandb_name = parse_config()
    set_seed(config.seed)
    init_wandb(config, wandb_name)
    trainer = GANTrainer(config)
    final_checkpoint = trainer.train()
    print(f"Training finished. Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
