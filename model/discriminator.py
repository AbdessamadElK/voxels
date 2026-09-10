import warnings
from functools import partial
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

_FFN_EXPANSION = 2.66


# ---------------------------------------------------------------------------
# Norm factory
# ---------------------------------------------------------------------------

def _norm_layer(norm: str, num_features: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(num_features)
    if norm == "instance":
        return nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
    return nn.Identity()


def _maybe_spectral(conv: nn.Conv2d, use_spectral_norm: bool) -> nn.Module:
    if use_spectral_norm:
        return nn.utils.spectral_norm(conv)
    return conv


# ---------------------------------------------------------------------------
# Weight initialisation (pix2pix-style)
# ---------------------------------------------------------------------------

def init_weights(module: nn.Module, init_type: str = "normal", gain: float = 0.02) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if init_type == "normal":
                nn.init.normal_(m.weight, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight, gain=gain)
            else:
                raise ValueError(f"Unknown init_type: {init_type!r}")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.normal_(m.weight, 1.0, gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# NLayerDiscriminator  (PatchGAN, 70-px receptive field at n_layers=3)
# ---------------------------------------------------------------------------

class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator.

    Output is a 2-D score map; never applies sigmoid (loss owns the activation).
    When return_interm_feats=True, forward returns (score_map, [feat_0, ..., feat_n]).
    """

    def __init__(
        self,
        in_channels: int,
        ndf: int = 64,
        n_layers: int = 3,
        norm: str = "instance",
        use_spectral_norm: bool = False,
        kernel_size: int = 4,
        padding: int = 1,
        leaky_slope: float = 0.2,
        bias: bool = False,
        return_interm_feats: bool = False,
    ) -> None:
        super().__init__()
        self.return_interm_feats = return_interm_feats

        mk = partial(_maybe_spectral, use_spectral_norm=use_spectral_norm)
        kw, pw = kernel_size, padding

        layers: list[nn.Module] = []

        # First layer — no norm
        layers.append(mk(nn.Conv2d(in_channels, ndf, kw, stride=2, padding=pw, bias=True)))
        layers.append(nn.LeakyReLU(leaky_slope, inplace=True))

        # Middle layers
        mult = 1
        for n in range(1, n_layers):
            mult_prev = mult
            mult = min(2 ** n, 8)
            layers.append(mk(nn.Conv2d(ndf * mult_prev, ndf * mult, kw, stride=2, padding=pw, bias=bias)))
            layers.append(_norm_layer(norm, ndf * mult))
            layers.append(nn.LeakyReLU(leaky_slope, inplace=True))

        # Stride-1 penultimate layer
        mult_prev = mult
        mult = min(2 ** n_layers, 8)
        layers.append(mk(nn.Conv2d(ndf * mult_prev, ndf * mult, kw, stride=1, padding=pw, bias=bias)))
        layers.append(_norm_layer(norm, ndf * mult))
        layers.append(nn.LeakyReLU(leaky_slope, inplace=True))

        # Final 1-channel output
        layers.append(mk(nn.Conv2d(ndf * mult, 1, kw, stride=1, padding=pw, bias=True)))

        self.layers = nn.ModuleList(layers)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        # x: (B, C, H, W)
        feats: list[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x)
            if self.return_interm_feats and isinstance(layer, nn.LeakyReLU):
                feats.append(x)
        if self.return_interm_feats:
            return x, feats
        return x


# ---------------------------------------------------------------------------
# PixelDiscriminator  (1×1 PatchGAN)
# ---------------------------------------------------------------------------

class PixelDiscriminator(nn.Module):
    """Pixel-wise discriminator: three 1×1 convolutions."""

    def __init__(
        self,
        in_channels: int,
        ndf: int = 64,
        norm: str = "instance",
        use_spectral_norm: bool = False,
        leaky_slope: float = 0.2,
        bias: bool = False,
        return_interm_feats: bool = False,
    ) -> None:
        super().__init__()
        self.return_interm_feats = return_interm_feats
        mk = partial(_maybe_spectral, use_spectral_norm=use_spectral_norm)

        self.conv0 = mk(nn.Conv2d(in_channels, ndf, 1, stride=1, padding=0, bias=True))
        self.act0  = nn.LeakyReLU(leaky_slope, inplace=True)
        self.norm1 = _norm_layer(norm, ndf)
        self.conv1 = mk(nn.Conv2d(ndf, ndf * 2, 1, stride=1, padding=0, bias=bias))
        self.act1  = nn.LeakyReLU(leaky_slope, inplace=True)
        self.conv2 = mk(nn.Conv2d(ndf * 2, 1, 1, stride=1, padding=0, bias=True))

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        feats: list[torch.Tensor] = []
        x = self.act0(self.conv0(x))
        if self.return_interm_feats:
            feats.append(x)
        x = self.act1(self.norm1(self.conv1(x)))
        if self.return_interm_feats:
            feats.append(x)
        x = self.conv2(x)
        if self.return_interm_feats:
            return x, feats
        return x


# ---------------------------------------------------------------------------
# MultiscaleDiscriminator
# ---------------------------------------------------------------------------

class MultiscaleDiscriminator(nn.Module):
    """Stack of num_D NLayerDiscriminators operating at successive half-scales.

    Forward returns a list of per-scale outputs (or (output, feats) pairs when
    return_interm_feats=True).
    """

    def __init__(
        self,
        in_channels: int,
        ndf: int = 64,
        n_layers: int = 3,
        norm: str = "instance",
        use_spectral_norm: bool = False,
        kernel_size: int = 4,
        padding: int = 1,
        leaky_slope: float = 0.2,
        bias: bool = False,
        num_D: int = 2,
        return_interm_feats: bool = False,
    ) -> None:
        super().__init__()
        self.return_interm_feats = return_interm_feats
        self.discriminators = nn.ModuleList([
            NLayerDiscriminator(
                in_channels=in_channels,
                ndf=ndf,
                n_layers=n_layers,
                norm=norm,
                use_spectral_norm=use_spectral_norm,
                kernel_size=kernel_size,
                padding=padding,
                leaky_slope=leaky_slope,
                bias=bias,
                return_interm_feats=return_interm_feats,
            )
            for _ in range(num_D)
        ])

    def forward(
        self, x: torch.Tensor
    ) -> list:
        # Returns list length num_D; each element is a score map or (score, feats) tuple.
        outputs = []
        inp = x
        for i, D in enumerate(self.discriminators):
            if i > 0:
                inp = F.avg_pool2d(inp, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            outputs.append(D(inp))
        return outputs


# ---------------------------------------------------------------------------
# TemporalBlock — pre-LN MHSA + GELU FFN over the sequence (temporal) dim
# ---------------------------------------------------------------------------

class TemporalBlock(nn.Module):
    """Standard pre-LN transformer encoder block operating on (N, L, d) sequences.

    Uses explicit Q/K/V projections with manual scaled-dot-product attention so
    that autograd.grad(create_graph=True) works for WGAN-GP gradient penalty.
    All projections and FFN linears are wrapped with spectral_norm.
    """

    def __init__(
        self,
        d: int,
        heads: int,
        ffn_expansion: float = _FFN_EXPANSION,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert d % heads == 0, f"d={d} must be divisible by heads={heads}"
        self.heads    = heads
        self.head_dim = d // heads
        self.scale    = self.head_dim ** -0.5

        self.norm1    = nn.LayerNorm(d)
        self.q_proj   = nn.utils.spectral_norm(nn.Linear(d, d, bias=bias))
        self.k_proj   = nn.utils.spectral_norm(nn.Linear(d, d, bias=bias))
        self.v_proj   = nn.utils.spectral_norm(nn.Linear(d, d, bias=bias))
        self.out_proj = nn.utils.spectral_norm(nn.Linear(d, d, bias=bias))

        self.norm2 = nn.LayerNorm(d)
        hidden = int(d * ffn_expansion)
        self.ffn = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(d, hidden, bias=bias)),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Linear(hidden, d, bias=bias)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, L, d)
        N, L, _ = x.shape
        n = self.norm1(x)
        q = self.q_proj(n).view(N, L, self.heads, self.head_dim).transpose(1, 2)  # (N, h, L, hd)
        k = self.k_proj(n).view(N, L, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(n).view(N, L, self.heads, self.head_dim).transpose(1, 2)
        w = (q @ k.transpose(-2, -1)) * self.scale                                # (N, h, L, L)
        a = (F.softmax(w, dim=-1) @ v).transpose(1, 2).reshape(N, L, -1)          # (N, L, d)
        x = x + self.out_proj(a)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# PixelTemporalCritic — per-super-pixel temporal-attention discriminator
# ---------------------------------------------------------------------------

class PixelTemporalCritic(nn.Module):
    """Temporal-attention discriminator treating each spatial super-pixel independently.

    Splits the concatenated input into a conditioning frame and a target voxel.
    The target's T channels become temporal tokens; a Conv3d stem with temporal
    kernel=1 downsamples spatially only, preserving the T=15 time axis intact.
    No activation is applied before the spatial stem — the signed voxel values
    enter the Conv3d directly.

    Input : (B, C_cond + C_target, H, W)  if conditional
            (B, C_target,          H, W)  otherwise
    Output: (B, 1, H', W') raw logit map, no sigmoid.
    H' = floor((H - s) / s) + 1,  W' = floor((W - s) / s) + 1.
    """

    def __init__(
        self,
        in_channels_cond:   int   = 15,
        in_channels_target: int   = 15,
        conditional:        bool  = True,
        d:                  int   = 64,
        L:                  int   = 4,
        s:                  int   = 8,
        heads:              int   = 4,
        ffn_expansion:      float = _FFN_EXPANSION,
        bias:               bool  = False,
        use_projection:     bool  = False,
        return_interm_feats: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels_cond    = in_channels_cond
        self.in_channels_target  = in_channels_target
        self.conditional         = conditional
        self.use_projection      = use_projection
        self.return_interm_feats = return_interm_feats

        # Spatial stem: (B, 1, T, H, W) -> (B, d, T, H', W')
        # kernel=(1, s, s) leaves the temporal axis length unchanged.
        self.spatial_stem = nn.utils.spectral_norm(
            nn.Conv3d(1, d, kernel_size=(1, s, s), stride=(1, s, s), bias=bias)
        )

        # Conditioning stem: (B, C_cond, H, W) -> (B, d, H', W')
        if conditional:
            self.cond_stem = nn.utils.spectral_norm(
                nn.Conv2d(in_channels_cond, d, kernel_size=s, stride=s, bias=bias)
            )

        # Temporal positional embedding (T slots) and CLS token.
        # trunc_normal init (std=0.02) following ViT: gives D non-zero output at step 0,
        # avoiding the zero-attractor caused by zeros+spectral_norm+zero_bias.
        self.pos_emb   = nn.Parameter(torch.empty(in_channels_target, d))
        self.cls_token = nn.Parameter(torch.empty(1, 1, d))
        nn.init.trunc_normal_(self.pos_emb,   std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Encoder
        self.encoder = nn.ModuleList([
            TemporalBlock(d, heads, ffn_expansion, bias) for _ in range(L)
        ])

        # Head: CLS feature -> scalar logit
        self.head = nn.utils.spectral_norm(nn.Linear(d, 1, bias=True))

        # Optional CLS·conditioning projection term
        if use_projection:
            self.proj = nn.utils.spectral_norm(nn.Linear(d, d, bias=False))

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        # x: (B, C_cond + C_target, H, W) or (B, C_target, H, W)
        B = x.shape[0]

        if self.conditional:
            cond   = x[:, :self.in_channels_cond]   # (B, C_cond, H, W)
            target = x[:, self.in_channels_cond:]   # (B, C_target, H, W)
        else:
            cond   = None
            target = x

        # --- Spatial stem ---
        # (B, C_target, H, W) -> (B, 1, T, H, W) -> (B, d, T, H', W')
        stem = self.spatial_stem(target.unsqueeze(1))
        _, d, T, Hp, Wp = stem.shape                                # T == in_channels_target

        # Flatten spatial: (B, d, T, H', W') -> (B, H', W', T, d) -> (B*H'*W', T, d)
        tokens = stem.permute(0, 3, 4, 2, 1).reshape(B * Hp * Wp, T, d)

        # --- Temporal positional embedding ---
        tokens = tokens + self.pos_emb                              # (B*H'*W', T, d)

        # --- Conditioning ---
        cond_vec = None
        if self.conditional and cond is not None:
            cv       = self.cond_stem(cond)                         # (B, d, H', W')
            cond_vec = cv.permute(0, 2, 3, 1).reshape(B * Hp * Wp, 1, d)
            tokens   = tokens + cond_vec                            # broadcast over T

        # --- Prepend CLS ---
        cls    = self.cls_token.expand(B * Hp * Wp, -1, -1)        # (B*H'*W', 1, d)
        tokens = torch.cat([cls, tokens], dim=1)                    # (B*H'*W', T+1, d)

        # --- Encoder ---
        feats: list[torch.Tensor] = []
        for block in self.encoder:
            tokens = block(tokens)
            if self.return_interm_feats:
                feats.append(tokens[:, 0])                          # CLS at each depth

        # --- Head ---
        cls_out = tokens[:, 0]                                      # (B*H'*W', d)
        logit   = self.head(cls_out)                                # (B*H'*W', 1)

        if self.use_projection and cond_vec is not None:
            proj_cond = self.proj(cond_vec.squeeze(1))              # (B*H'*W', d)
            logit = logit + (cls_out * proj_cond).sum(-1, keepdim=True)

        # (B*H'*W', 1) -> (B, 1, H', W')
        score = logit.reshape(B, Hp, Wp).unsqueeze(1)

        if self.return_interm_feats:
            return score, feats
        return score


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_discriminator(
    netD: str = "basic",
    in_channels_cond: int = 15,
    in_channels_target: int = 15,
    conditional: bool = True,
    ndf: int = 64,
    n_layers: int = 3,
    num_D: int = 2,
    norm: str = "instance",
    use_spectral_norm: bool = False,
    kernel_size: int = 4,
    padding: int = 1,
    leaky_slope: float = 0.2,
    return_interm_feats: bool = False,
    init_type: str = "normal",
    init_gain: float = 0.02,
    gan_mode: str = "lsgan",
    # PixelTemporalCritic-only
    temporal_L: int = 4,
    temporal_s: int = 8,
    use_projection: bool = False,
) -> nn.Module:
    """Instantiate and weight-initialise a discriminator.

    When conditional=True the first conv receives in_channels_cond + in_channels_target
    channels; otherwise it receives in_channels_target only.

    Passing norm='batch' with gan_mode='wgangp' overrides norm to 'instance' (BatchNorm
    is incompatible with the per-sample gradient penalty).
    """
    if norm == "batch" and gan_mode == "wgangp":
        warnings.warn(
            "norm='batch' is incompatible with wgangp (per-sample gradient penalty). "
            "Overriding to norm='instance'.",
            stacklevel=2,
        )
        norm = "instance"

    in_ch = (in_channels_cond + in_channels_target) if conditional else in_channels_target

    shared_kw = dict(
        in_channels=in_ch,
        ndf=ndf,
        norm=norm,
        use_spectral_norm=use_spectral_norm,
        leaky_slope=leaky_slope,
        return_interm_feats=return_interm_feats,
    )

    if netD in ("basic", "nlayer"):
        net = NLayerDiscriminator(
            **shared_kw,
            n_layers=n_layers,
            kernel_size=kernel_size,
            padding=padding,
        )
    elif netD == "pixel":
        net = PixelDiscriminator(**shared_kw)
    elif netD == "multiscale":
        net = MultiscaleDiscriminator(
            **shared_kw,
            n_layers=n_layers,
            kernel_size=kernel_size,
            padding=padding,
            num_D=num_D,
        )
    elif netD == "temporal":
        net = PixelTemporalCritic(
            in_channels_cond=in_channels_cond,
            in_channels_target=in_channels_target,
            conditional=conditional,
            d=ndf,
            L=temporal_L,
            s=temporal_s,
            use_projection=use_projection,
            return_interm_feats=return_interm_feats,
        )
        # PixelTemporalCritic initialises its own parameters; skip the pix2pix init_weights
        # pass which only covers Conv2d/ConvTranspose2d and would miss Conv3d/Linear.
        return net
    else:
        raise ValueError(
            f"Unknown netD: {netD!r}. Choose from 'basic', 'pixel', 'multiscale', 'temporal'."
        )

    init_weights(net, init_type=init_type, gain=init_gain)
    return net
