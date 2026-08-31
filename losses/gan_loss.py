from typing import Union

import torch
import torch.nn as nn
import torch.autograd as autograd


# ---------------------------------------------------------------------------
# GANLoss
# ---------------------------------------------------------------------------

class GANLoss(nn.Module):
    """Adversarial loss supporting vanilla, lsgan, hinge, and wgangp modes.

    Call signature:
        loss = criterion(pred, target_is_real, for_discriminator)

    pred may be a single tensor or a list of tensors (multiscale); losses are summed.
    Never apply sigmoid here — the discriminator output is raw logits.
    """

    REAL_LABEL: float = 1.0
    FAKE_LABEL: float = 0.0

    def __init__(
        self,
        gan_mode: str = "lsgan",
        real_label: float = 1.0,
        fake_label: float = 0.0,
    ) -> None:
        super().__init__()
        if gan_mode not in ("vanilla", "lsgan", "hinge", "wgangp"):
            raise ValueError(f"Unknown gan_mode: {gan_mode!r}")
        self.gan_mode   = gan_mode
        self.real_label = real_label
        self.fake_label = fake_label

        if gan_mode == "vanilla":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif gan_mode == "lsgan":
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _target_tensor(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        val = self.real_label if target_is_real else self.fake_label
        return pred.new_full(pred.shape, val)

    def _loss_single(
        self,
        pred: torch.Tensor,
        target_is_real: bool,
        for_discriminator: bool,
    ) -> torch.Tensor:
        if self.gan_mode in ("vanilla", "lsgan"):
            target = self._target_tensor(pred, target_is_real)
            return self.loss_fn(pred, target)

        if self.gan_mode == "hinge":
            if for_discriminator:
                if target_is_real:
                    return torch.relu(1.0 - pred).mean()
                return torch.relu(1.0 + pred).mean()
            # generator: maximise D(fake) -> minimise -D(fake)
            return -pred.mean()

        # wgangp
        if for_discriminator:
            if target_is_real:
                return -pred.mean()
            return pred.mean()
        return -pred.mean()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: Union[torch.Tensor, list],
        target_is_real: bool,
        for_discriminator: bool,
    ) -> torch.Tensor:
        if isinstance(pred, list):
            # multiscale: each element may be a tensor or (tensor, feats) tuple
            total = pred[0].new_zeros(1).squeeze()
            for p in pred:
                if isinstance(p, tuple):
                    p = p[0]
                total = total + self._loss_single(p, target_is_real, for_discriminator)
            return total
        if isinstance(pred, tuple):
            pred = pred[0]
        return self._loss_single(pred, target_is_real, for_discriminator)


# ---------------------------------------------------------------------------
# Gradient penalty (WGAN-GP)
# ---------------------------------------------------------------------------

def gradient_penalty(
    netD: nn.Module,
    real_target: torch.Tensor,
    fake_target: torch.Tensor,
    cond: torch.Tensor | None = None,
    lambda_gp: float = 10.0,
    conditional: bool = True,
) -> torch.Tensor:
    """Compute WGAN-GP gradient penalty.

    Interpolates between real_target and fake_target with a per-sample uniform
    alpha, builds the discriminator input (optionally prepending cond), runs the
    critic, and returns lambda_gp * E[(||grad|| - 1)^2].

    For multiscale discriminators the penalty is summed over scales.
    """
    B = real_target.size(0)
    # (B, 1, 1, 1) uniform alpha
    alpha = torch.rand(B, 1, 1, 1, device=real_target.device, dtype=real_target.dtype)
    interp = (alpha * real_target + (1.0 - alpha) * fake_target).requires_grad_(True)

    if conditional and cond is not None:
        d_input = torch.cat([cond, interp], dim=1)
    else:
        d_input = interp

    d_out = netD(d_input)

    # Flatten multiscale / feats-tuple outputs to a list of score tensors
    if isinstance(d_out, list):
        scores = [o[0] if isinstance(o, tuple) else o for o in d_out]
    else:
        scores = [d_out[0] if isinstance(d_out, tuple) else d_out]

    penalty = real_target.new_zeros(1).squeeze()
    for score in scores:
        ones = torch.ones_like(score)
        grads = autograd.grad(
            outputs=score,
            inputs=interp,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        # grads: (B, C, H, W) -> norm over C, H, W
        grad_norm = grads.flatten(1).norm(2, dim=1)          # (B,)
        penalty = penalty + ((grad_norm - 1.0) ** 2).mean()

    return lambda_gp * penalty
