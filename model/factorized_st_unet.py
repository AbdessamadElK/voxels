import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_DIM = 16
# [encoder level 1, encoder level 2, bottleneck]
DEFAULT_NUM_BLOCKS = (1, 2, 2)
DEFAULT_NUM_HEADS = (1, 2, 4)
FFN_EXPANSION = 2.66
SPATIAL_KERNEL = 3
TEMPORAL_KERNEL = 3
GROUP_CANDIDATES = (32, 16, 8, 4, 2, 1)
LEVEL_WIDTHS = (1, 2, 4)          # dim multipliers at H, H/2, H/4

# Token axes of a (B, C, T, H, W) volume.
AXIS_DIMS = {"t": (2,), "h": (3,), "w": (4,), "hw": (3, 4)}
VOLUME_DIMS = (2, 3, 4)
# Divided space-time attends over time then over the whole frame; axial splits the
# frame into its two spatial axes, which keeps every attention matrix one-dimensional.
ATTENTION_AXES = {"divided": ("t", "hw"), "axial": ("t", "h", "w")}


def _norm_groups(dim: int) -> int:
    for groups in GROUP_CANDIDATES:
        if dim % groups == 0:
            return groups
    return 1


def _resize(x: torch.Tensor, size: torch.Size) -> torch.Tensor:
    """Resample the spatial axes of a (B, C, T, H, W) volume to size.

    size carries (T, H, W) and keeps T unchanged, so the temporal axis maps onto
    itself and only H and W move.
    """
    return F.interpolate(x, size=tuple(size), mode="trilinear", align_corners=False)


def _to_tokens(x: torch.Tensor, axis: str) -> tuple[torch.Tensor, tuple[int, ...], torch.Size]:
    """Fold a (B, C, T, H, W) volume into (N, L, C) tokens running along axis.

    Returns the tokens with the permutation and folded shape needed to undo the fold.
    """
    token_dims = AXIS_DIMS[axis]
    batch_dims = tuple(d for d in VOLUME_DIMS if d not in token_dims)
    perm = (0, *batch_dims, *token_dims, 1)
    folded = x.permute(perm)                                   # (B, *batch, *tokens, C)
    length = 1
    for dim in token_dims:
        length *= x.shape[dim]
    return folded.reshape(-1, length, x.shape[1]), perm, folded.shape


def _from_tokens(
    tokens: torch.Tensor, perm: tuple[int, ...], folded_shape: torch.Size
) -> torch.Tensor:
    """Invert _to_tokens, restoring (B, C, T, H, W)."""
    inverse = [0] * len(perm)
    for position, dim in enumerate(perm):
        inverse[dim] = position
    return tokens.reshape(folded_shape).permute(inverse)


class AxisAttention(nn.Module):
    """Pre-LayerNorm self-attention along one axis of a (B, C, T, H, W) volume.

    Every axis outside the token axis folds into the batch, so an axis of length L
    costs L squared attention instead of (T*H*W) squared.
    scaled_dot_product_attention keeps the L by L matrix off memory.
    """

    def __init__(self, dim: int, num_heads: int, axis: str, bias: bool = False) -> None:
        super().__init__()
        if axis not in AXIS_DIMS:
            raise ValueError(f"Unknown attention axis {axis!r}. Available: {sorted(AXIS_DIMS)}")
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")

        self.axis = axis
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self._init_weights()

    def _init_weights(self) -> None:
        for linear in (self.qkv, self.proj):
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, perm, folded_shape = _to_tokens(x, self.axis)  # (N, L, C)
        N, L, C = tokens.shape

        qkv = self.qkv(self.norm(tokens))                      # (N, L, 3C)
        qkv = qkv.reshape(N, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)         # (N, heads, L, head_dim)

        attended = F.scaled_dot_product_attention(q, k, v)     # (N, heads, L, head_dim)
        attended = attended.transpose(1, 2).reshape(N, L, C)   # (N, L, C)

        return _from_tokens(tokens + self.proj(attended), perm, folded_shape)


class FactorizedConvBlock(nn.Module):
    """Residual R(2+1)D block: spatial (1, k, k) conv, GELU, temporal (t, 1, 1) conv.

    Padding holds every axis, so T and the spatial size come out as they went in.
    """

    def __init__(
        self,
        dim: int,
        spatial_kernel: int = SPATIAL_KERNEL,
        temporal_kernel: int = TEMPORAL_KERNEL,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(_norm_groups(dim), dim),
            nn.Conv3d(
                dim, dim,
                kernel_size=(1, spatial_kernel, spatial_kernel),
                padding=(0, spatial_kernel // 2, spatial_kernel // 2),
                bias=bias,
            ),
            nn.GELU(),
            nn.Conv3d(
                dim, dim,
                kernel_size=(temporal_kernel, 1, 1),
                padding=(temporal_kernel // 2, 0, 0),
                bias=bias,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)                               # (B, C, T, H, W)


class FactorizedAttentionBlock(nn.Module):
    """Self-attention along each axis of attn_type in turn, then a pointwise FFN.

    The axis sequence carries the cross-axis interaction: within one block a token
    meets every frame of its own pixel column and every pixel of its own frame.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_type: str = "divided",
        ffn_expansion_factor: float = FFN_EXPANSION,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if attn_type not in ATTENTION_AXES:
            raise ValueError(
                f"Unknown attn_type {attn_type!r}. Available: {sorted(ATTENTION_AXES)}"
            )
        self.attentions = nn.ModuleList(
            AxisAttention(dim, num_heads, axis, bias=bias)
            for axis in ATTENTION_AXES[attn_type]
        )
        hidden = int(dim * ffn_expansion_factor)
        self.norm = nn.GroupNorm(_norm_groups(dim), dim)
        self.ffn = nn.Sequential(
            nn.Conv3d(dim, hidden, 1, bias=bias),
            nn.GELU(),
            nn.Conv3d(hidden, dim, 1, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attention in self.attentions:
            x = attention(x)                                   # (B, C, T, H, W)
        return x + self.ffn(self.norm(x))


class FactorizedSTUNet(nn.Module):
    """Space-time UNet that treats the 15 event time bins as a volume, not as channels.

    A stem lifts (B, T, H, W) to a (B, dim, T, H, W) volume. Factorized convolutions
    and factorized attention carry it through two spatial downsamples and back; T
    never changes, and only H and W are reduced. Every block splits its work by axis:
    R(2+1)D convolutions run spatially then temporally, and the bottleneck attention
    runs along one axis at a time, which is what makes attention over a T*H*W volume
    affordable at all.

    Args:
        in_channels: time bins T, carried in the channel axis of the input.
        dim: width at full resolution; levels widen by LEVEL_WIDTHS.
        num_blocks: block counts for encoder level 1, encoder level 2, bottleneck.
        num_heads: attention heads per level; index 2 drives the bottleneck.
        attn_type: divided or axial.
        decoder_attn: add one attention block at decoder level 2 (H/2).
    """

    def __init__(
        self,
        in_channels: int = 15,
        dim: int = DEFAULT_DIM,
        num_blocks: list[int] = None,
        num_heads: list[int] = None,
        attn_type: str = "divided",
        decoder_attn: bool = False,
        spatial_kernel: int = SPATIAL_KERNEL,
        temporal_kernel: int = TEMPORAL_KERNEL,
        ffn_expansion_factor: float = FFN_EXPANSION,
        bias: bool = False,
    ) -> None:
        super().__init__()
        num_blocks = list(DEFAULT_NUM_BLOCKS if num_blocks is None else num_blocks)
        num_heads = list(DEFAULT_NUM_HEADS if num_heads is None else num_heads)
        if len(num_blocks) < len(LEVEL_WIDTHS):
            raise ValueError(f"num_blocks needs {len(LEVEL_WIDTHS)} entries, got {num_blocks}")
        if len(num_heads) < len(LEVEL_WIDTHS):
            raise ValueError(f"num_heads needs {len(LEVEL_WIDTHS)} entries, got {num_heads}")

        self.num_bins = in_channels

        width1, width2, width3 = (dim * w for w in LEVEL_WIDTHS)
        conv_kw = dict(
            spatial_kernel=spatial_kernel, temporal_kernel=temporal_kernel, bias=bias
        )
        attn_kw = dict(
            attn_type=attn_type, ffn_expansion_factor=ffn_expansion_factor, bias=bias
        )

        # (B, 1, T, H, W) -> (B, dim, T, H, W). Spatial only; the encoder blocks mix time.
        self.stem = nn.Conv3d(
            1, width1,
            kernel_size=(1, spatial_kernel, spatial_kernel),
            padding=(0, spatial_kernel // 2, spatial_kernel // 2),
            bias=bias,
        )

        # Encoder
        self.encoder_level1 = nn.Sequential(
            *[FactorizedConvBlock(width1, **conv_kw) for _ in range(num_blocks[0])]
        )
        # (B, dim, T, H, W) -> (B, dim*2, T, H/2, W/2)
        self.down1 = nn.Conv3d(
            width1, width2,
            kernel_size=(1, spatial_kernel, spatial_kernel),
            stride=(1, 2, 2),
            padding=(0, spatial_kernel // 2, spatial_kernel // 2),
            bias=bias,
        )
        self.encoder_level2 = nn.Sequential(
            *[FactorizedConvBlock(width2, **conv_kw) for _ in range(num_blocks[1])]
        )
        # (B, dim*2, T, H/2, W/2) -> (B, dim*4, T, H/4, W/4)
        self.down2 = nn.Conv3d(
            width2, width3,
            kernel_size=(1, spatial_kernel, spatial_kernel),
            stride=(1, 2, 2),
            padding=(0, spatial_kernel // 2, spatial_kernel // 2),
            bias=bias,
        )

        # Bottleneck — the only level where attention runs by default.
        self.latent = nn.Sequential(
            *[
                FactorizedAttentionBlock(width3, num_heads[2], **attn_kw)
                for _ in range(num_blocks[2])
            ]
        )

        # Decoder level 2, at H/2
        # (B, dim*4, T, H/4, W/4) -> (B, dim*2, T, H/2, W/2)
        self.up2 = nn.Conv3d(
            width3, width2,
            kernel_size=(1, spatial_kernel, spatial_kernel),
            padding=(0, spatial_kernel // 2, spatial_kernel // 2),
            bias=bias,
        )
        # after concat with the skip: dim*2 + dim*2 = dim*4
        decoder2 = [FactorizedConvBlock(width3, **conv_kw) for _ in range(num_blocks[1])]
        if decoder_attn:
            decoder2.insert(0, FactorizedAttentionBlock(width3, num_heads[1], **attn_kw))
        self.decoder_level2 = nn.Sequential(*decoder2)

        # Decoder level 1, at full resolution
        # (B, dim*4, T, H/2, W/2) -> (B, dim, T, H, W)
        self.up1 = nn.Conv3d(
            width3, width1,
            kernel_size=(1, spatial_kernel, spatial_kernel),
            padding=(0, spatial_kernel // 2, spatial_kernel // 2),
            bias=bias,
        )
        # after concat with the skip: dim + dim = dim*2
        self.decoder_level1 = nn.Sequential(
            *[FactorizedConvBlock(width2, **conv_kw) for _ in range(num_blocks[0])]
        )

        # (B, dim*2, T, H, W) -> (B, 1, T, H, W)
        self.output_proj = nn.Conv3d(
            width2, 1,
            kernel_size=(1, spatial_kernel, spatial_kernel),
            padding=(0, spatial_kernel // 2, spatial_kernel // 2),
            bias=bias,
        )

        self._init_conv_weights()

    def _init_conv_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # fan_out counts a single output channel here, so kaiming leaves the output at
        # 88x the ground-truth scale and the first steps go to shrinking it. Xavier puts
        # it at 12x, next to UNetTransformer's 17x, so the two start comparable.
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W) -> (B, T, H, W)
        if x.shape[1] != self.num_bins:
            raise ValueError(
                f"Expected {self.num_bins} time bins on dim 1, got shape {tuple(x.shape)}"
            )

        volume = x.unsqueeze(1)                                    # (B, 1, T, H, W)
        feat = self.stem(volume)                                   # (B, dim, T, H, W)

        enc1 = self.encoder_level1(feat)                           # (B, dim, T, H, W)
        enc2 = self.encoder_level2(self.down1(enc1))               # (B, dim*2, T, H/2, W/2)
        latent = self.latent(self.down2(enc2))                     # (B, dim*4, T, H/4, W/4)

        up2 = self.up2(_resize(latent, enc2.shape[2:]))            # (B, dim*2, T, H/2, W/2)
        dec2 = self.decoder_level2(
            torch.cat([up2, enc2], dim=1)                          # (B, dim*4, T, H/2, W/2)
        )

        up1 = self.up1(_resize(dec2, enc1.shape[2:]))              # (B, dim, T, H, W)
        dec1 = self.decoder_level1(
            torch.cat([up1, enc1], dim=1)                          # (B, dim*2, T, H, W)
        )

        return self.output_proj(dec1).squeeze(1)                   # (B, T, H, W)
