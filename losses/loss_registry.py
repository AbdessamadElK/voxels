import inspect
import logging

import torch.nn as nn

from .v2ce_losses_v2 import CombinedLoss

LOSS_KEY = "loss_fn"


class LossRegistry:
    """Maps loss names to classes and builds them from config dicts."""

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, loss_class: type) -> None:
        """Register a loss class under a name, replacing any earlier entry."""
        if not issubclass(loss_class, nn.Module):
            raise TypeError(f"{loss_class.__name__} is not an nn.Module subclass")
        cls._registry[name] = loss_class

    @classmethod
    def get(cls, name: str) -> type:
        """Return the class registered under name."""
        if name not in cls._registry:
            raise KeyError(f"Unknown loss '{name}'. Available: {cls.list_available()}")
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> list[str]:
        """Return the registered loss names, sorted."""
        return sorted(cls._registry)

    @classmethod
    def build(cls, config: dict, logger: logging.Logger | None = None) -> nn.Module:
        """Instantiate the loss named by config['loss_fn'].

        Keys the loss does not accept are skipped and reported, which keeps configs
        that carry weights for other loss variants usable.
        """
        if LOSS_KEY not in config:
            raise KeyError(
                f"Loss config is missing '{LOSS_KEY}'. Available: {cls.list_available()}"
            )

        name = config[LOSS_KEY]
        loss_class = cls.get(name)

        kwargs = {k: v for k, v in config.items() if k != LOSS_KEY}
        accepted = set(inspect.signature(loss_class.__init__).parameters) - {"self"}
        skipped = sorted(set(kwargs) - accepted)
        if skipped and logger is not None:
            logger.warning(f"Loss '{name}' does not accept {skipped}; keys ignored.")

        return loss_class(**{k: v for k, v in kwargs.items() if k in accepted})


LossRegistry.register("CombinedLoss", CombinedLoss)
