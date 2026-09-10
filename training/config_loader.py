import json
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar

# Runtime knobs a CLI flag may override. Architecture and loss weights stay in JSON.
OVERRIDABLE_KEYS = frozenset(
    {
        "num_steps",
        "checkpoint_dir",
        "lr",
        "seed",
        "validate",
        "use_wandb",
        "wandb_project",
        "wandb_name",
        "model_path",
        "continue_training",
        "device",
        "batch_size",
    }
)

DEFAULT_NUM_STEPS = 200_000
DEFAULT_CHECKPOINT_DIR = "checkpoints"


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or incomplete."""


@dataclass
class _BaseConfig:
    """Declared fields plus an `extra` bag for keys the JSON adds beyond them.

    Extra keys read back as plain attributes, so `config.dim` works whether `dim` is
    declared or not.
    """

    extra: dict[str, Any] = field(default_factory=dict)

    # Keys a JSON file must define; subclasses narrow this.
    REQUIRED: ClassVar[frozenset[str]] = frozenset()

    def __getattr__(self, name: str) -> Any:
        # Reached only when normal attribute lookup fails.
        if name != "extra":
            try:
                return self.__dict__["extra"][name]
            except KeyError:
                pass
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

    def __contains__(self, name: str) -> bool:
        return name in self.as_dict()

    def get(self, name: str, default: Any = None) -> Any:
        return self.as_dict().get(name, default)

    def as_dict(self) -> dict[str, Any]:
        """Flatten declared fields and extras into one dict."""
        declared = {
            f.name: getattr(self, f.name) for f in fields(self) if f.name != "extra"
        }
        return {**declared, **self.extra}

    @classmethod
    def from_dict(cls, values: dict[str, Any], source: str) -> "_BaseConfig":
        missing = sorted(cls.REQUIRED.difference(values))
        if missing:
            raise ConfigError(f"'{source}' is missing required keys: {missing}")

        declared = {f.name for f in fields(cls) if f.name != "extra"}
        known = {k: v for k, v in values.items() if k in declared}
        extra = {k: v for k, v in values.items() if k not in declared}
        return cls(**known, extra=extra)


@dataclass
class ModelConfig(_BaseConfig):
    backbone:     str = ""
    in_channels:  int = 15
    out_channels: int = 15

    REQUIRED = frozenset({"backbone", "in_channels", "out_channels"})


@dataclass
class LossConfig(_BaseConfig):
    loss_fn:    str   = "CombinedLoss"
    lambda_stp: float = 1.0
    lambda_tp:  float = 1.0
    lambda_ef:  float = 1.0
    lambda_ss:  float = 1.0
    lambda_ts:  float = 1.0
    tp_kwargs:  dict[str, Any] = field(default_factory=dict)

    REQUIRED = frozenset({"loss_fn"})


@dataclass
class TrainingConfig(_BaseConfig):
    batch_size:    int   = 6
    num_workers:   int   = 8
    lr:            float = 2e-4
    weight_decay:  float = 1e-4
    device:        str   = "cuda"
    seed:          int   = 1
    validate:      bool  = False
    use_wandb:     bool  = False
    wandb_project: str   = "EV_SNN"
    sum_freq:      int   = 100
    vis_freq:      int   = 1_000
    val_freq:      int   = 5_000
    save_freq:     int   = 10_000
    model_path:    str   = ""
    continue_training: bool = False

    # Runtime-only: absent from JSON by default, supplied by CLI or the fallbacks below.
    num_steps:      int = None
    checkpoint_dir: str = None

    REQUIRED = frozenset({"batch_size", "num_workers", "device", "seed"})


@dataclass
class FullConfig:
    model:    ModelConfig
    loss:     LossConfig
    training: TrainingConfig

    def as_dict(self) -> dict[str, Any]:
        return {
            "model":    self.model.as_dict(),
            "loss":     self.loss.as_dict(),
            "training": self.training.as_dict(),
        }


def load_json(path: str) -> dict[str, Any]:
    """Read a JSON object from path."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"Config file not found: '{path}'")
    try:
        with file_path.open(encoding="utf-8") as handle:
            values = json.load(handle)
    except json.JSONDecodeError as error:
        raise ConfigError(f"'{path}' is not valid JSON: {error}") from error
    if not isinstance(values, dict):
        raise ConfigError(f"'{path}' must hold a JSON object, got {type(values).__name__}")
    return values


def _coerce(value: Any, reference: Any) -> Any:
    """Convert value to the type of reference when the two disagree."""
    if reference is None or isinstance(value, type(reference)):
        return value
    if isinstance(reference, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "y")
        return bool(value)
    if isinstance(reference, (int, float, str)):
        return type(reference)(value)
    if isinstance(reference, list) and not isinstance(value, list):
        return [value]
    return value


def _apply_overrides(
    config: TrainingConfig,
    overrides: dict[str, Any],
    allowed: frozenset[str],
    logger: logging.Logger | None,
) -> None:
    """Write the overridable CLI values onto config, logging each change."""
    declared = {f.name for f in fields(config)}
    for key in sorted(overrides):
        if key not in allowed:
            continue
        new_value = overrides[key]
        if new_value is None:
            continue
        old_value = config.get(key)
        new_value = _coerce(new_value, old_value)
        if old_value == new_value:
            continue
        if key in declared:
            setattr(config, key, new_value)
        else:
            config.extra[key] = new_value
        if logger is not None:
            logger.info(f"CLI override: {key}: {old_value!r} -> {new_value!r}")


def _apply_runtime_defaults(
    config: TrainingConfig, logger: logging.Logger | None
) -> None:
    """Fill num_steps and checkpoint_dir when neither JSON nor CLI supplied them."""
    fallbacks = {
        "num_steps": DEFAULT_NUM_STEPS,
        "checkpoint_dir": DEFAULT_CHECKPOINT_DIR,
    }
    for key, fallback in fallbacks.items():
        if getattr(config, key) is None:
            setattr(config, key, fallback)
            if logger is not None:
                logger.info(f"Default applied: {key} = {fallback!r}")


def _report_ignored_overrides(
    overrides: dict[str, Any], allowed: frozenset[str], logger: logging.Logger | None
) -> None:
    ignored = sorted(set(overrides) - allowed)
    if ignored and logger is not None:
        logger.warning(
            f"CLI keys outside the overridable set were ignored: {ignored}. "
            f"Overridable keys: {sorted(allowed)}"
        )


def load_config(
    model_config_path:    str,
    loss_config_path:     str,
    training_config_path: str,
    cli_overrides:        dict[str, Any] = None,
    logger:               logging.Logger | None = None,
    extra_overridable:    frozenset[str] = frozenset(),
) -> FullConfig:
    """Load the three JSON configs and merge the CLI overrides into the training one.

    CLI values win over JSON, but only for the keys in OVERRIDABLE_KEYS plus
    extra_overridable; architecture and loss weights come from JSON alone. num_steps
    and checkpoint_dir fall back to module defaults when neither source provides them.
    """
    cli_overrides = cli_overrides or {}
    allowed = OVERRIDABLE_KEYS | frozenset(extra_overridable)

    model_values    = load_json(model_config_path)
    loss_values     = load_json(loss_config_path)
    training_values = load_json(training_config_path)

    if logger is not None:
        logger.info(f"Loaded model config    '{model_config_path}'")
        logger.info(f"Loaded loss config     '{loss_config_path}'")
        logger.info(f"Loaded training config '{training_config_path}'")

    model    = ModelConfig.from_dict(model_values, model_config_path)
    loss     = LossConfig.from_dict(loss_values, loss_config_path)
    training = TrainingConfig.from_dict(training_values, training_config_path)

    _report_ignored_overrides(cli_overrides, allowed, logger)
    _apply_overrides(training, cli_overrides, allowed, logger)
    _apply_runtime_defaults(training, logger)

    config = FullConfig(model=model, loss=loss, training=training)
    if logger is not None:
        logger.info(f"Merged configuration: {json.dumps(config.as_dict(), default=str)}")
    return config
