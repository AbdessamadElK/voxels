import inspect

import torch
import torch.nn as nn

from .factorized_st_unet import FactorizedSTUNet
from .swin_transformer import SwinEncoder, SwinTransformer
from .unet_transformer import UNetTransformer

BACKBONE_KEY = "backbone"


class BackboneRegistry:
    """Maps backbone names to model classes and builds them from config dicts."""

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, model_class: type) -> None:
        """Register a model class under a name, replacing any earlier entry."""
        if not issubclass(model_class, nn.Module):
            raise TypeError(f"{model_class.__name__} is not an nn.Module subclass")
        cls._registry[name] = model_class

    @classmethod
    def get(cls, name: str) -> type:
        """Return the class registered under name."""
        if name not in cls._registry:
            raise KeyError(
                f"Unknown backbone '{name}'. Available: {cls.list_available()}"
            )
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> list[str]:
        """Return the registered backbone names, sorted."""
        return sorted(cls._registry)

    @classmethod
    def build(cls, config: dict, device: str | torch.device = "cuda") -> nn.Module:
        """Instantiate the backbone named by config['backbone'] and move it to device.

        Every remaining key must name a constructor argument of that backbone, except
        out_channels, which a backbone tied to its input width may omit when the two
        match.
        """
        if BACKBONE_KEY not in config:
            raise KeyError(
                f"Model config is missing '{BACKBONE_KEY}'. "
                f"Available backbones: {cls.list_available()}"
            )

        name = config[BACKBONE_KEY]
        model_class = cls.get(name)

        kwargs = {k: v for k, v in config.items() if k != BACKBONE_KEY}
        parameters = inspect.signature(model_class.__init__).parameters
        accepted = set(parameters) - {"self"}
        cls._check_channel_width(name, kwargs, accepted)

        # A backbone with **kwargs absorbs superseded config keys instead of failing.
        takes_var_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )
        if takes_var_kwargs:
            return model_class(**kwargs).to(device)

        unsupported = sorted(set(kwargs) - accepted - {"out_channels"})
        if unsupported:
            raise TypeError(
                f"Backbone '{name}' does not accept {unsupported}. "
                f"Accepted keys: {sorted(accepted)}"
            )

        model = model_class(**{k: v for k, v in kwargs.items() if k in accepted})
        return model.to(device)

    @staticmethod
    def _check_channel_width(name: str, kwargs: dict, accepted: set[str]) -> None:
        """Reject an out_channels the backbone cannot honour."""
        if "out_channels" in accepted or "out_channels" not in kwargs:
            return
        if kwargs["out_channels"] != kwargs.get("in_channels"):
            raise ValueError(
                f"Backbone '{name}' produces in_channels outputs, so out_channels="
                f"{kwargs['out_channels']} cannot be honoured with in_channels="
                f"{kwargs.get('in_channels')}."
            )


BackboneRegistry.register("unet_transformer", UNetTransformer)
BackboneRegistry.register("swin_t7", SwinTransformer)
BackboneRegistry.register("factorized_st", FactorizedSTUNet)
