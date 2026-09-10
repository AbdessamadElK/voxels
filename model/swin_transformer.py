import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_decoder import SwinUNetDecoder

DEFAULT_PATCH_SIZE = 4
DEFAULT_EMBED_DIM = 96
DEFAULT_WINDOW_SIZE = 7
DEFAULT_MLP_RATIO = 4.0
DEFAULT_DEPTHS = [2, 2, 6, 2]
DEFAULT_NUM_HEADS = [3, 6, 12, 24]
DEFAULT_DROP_PATH_RATE = 0.1

TRUNC_NORMAL_STD = 0.02
MASK_FILL = -100.0
MERGE_FACTOR = 2

# Attention masks depend only on geometry, so cache them across blocks and steps.
_ATTN_MASK_CACHE: dict[tuple, torch.Tensor] = {}


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Split a feature map into non-overlapping windows."""
    # (B, H, W, C) -> (B*num_windows, window_size**2, C)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, window_size * window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Merge windows back into a feature map."""
    # (B*num_windows, window_size**2, C) -> (B, H, W, C)
    C = windows.shape[-1]
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, C)


def build_attention_mask(
    padded_h:    int,
    padded_w:    int,
    valid_h:     int,
    valid_w:     int,
    window_size: int,
    shift_size:  int,
    device:      torch.device,
) -> torch.Tensor | None:
    """Mask cross-region attention after a cyclic shift and attention onto padded tokens.

    Returns None when neither applies.
    """
    if shift_size == 0 and valid_h == padded_h and valid_w == padded_w:
        return None

    key = (padded_h, padded_w, valid_h, valid_w, window_size, shift_size, str(device))
    if key in _ATTN_MASK_CACHE:
        return _ATTN_MASK_CACHE[key]

    # Region ids separate the wrapped-around strips produced by torch.roll.
    region = torch.zeros((1, padded_h, padded_w, 1), device=device)
    if shift_size > 0:
        spans = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        region_id = 0
        for span_h in spans:
            for span_w in spans:
                region[:, span_h, span_w, :] = region_id
                region_id += 1

    valid = torch.zeros((1, padded_h, padded_w, 1), device=device)
    valid[:, :valid_h, :valid_w, :] = 1.0
    if shift_size > 0:
        valid = torch.roll(valid, shifts=(-shift_size, -shift_size), dims=(1, 2))

    region_windows = window_partition(region, window_size).squeeze(-1)   # (num_windows, N)
    valid_windows  = window_partition(valid, window_size).squeeze(-1)    # (num_windows, N)

    mask = region_windows.unsqueeze(1) - region_windows.unsqueeze(2)     # (num_windows, N, N)
    mask = mask.masked_fill(mask != 0, MASK_FILL).masked_fill(mask == 0, 0.0)
    mask = mask.masked_fill(valid_windows.unsqueeze(1) == 0, MASK_FILL)  # padded keys

    _ATTN_MASK_CACHE[key] = mask
    return mask


class DropPath(nn.Module):
    """Drop whole residual branches per sample during training."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        keep = x.new_empty(shape).bernoulli_(keep_prob)
        return x * keep / keep_prob


class Mlp(nn.Module):
    """Two-layer feed-forward network with GELU."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N, dim) -> (B, N, dim)
        return self.fc2(self.act(self.fc1(x)))


class WindowAttention(nn.Module):
    """Multi-head self-attention inside a window, with a relative position bias."""

    def __init__(
        self,
        dim:         int,
        window_size: int,
        num_heads:   int,
        qkv_bias:    bool = True,
        attn_drop:   float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")

        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        num_relative_positions = (2 * window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_positions, num_heads)
        )
        self.register_buffer(
            "relative_position_index",
            self._relative_position_index(window_size),
            persistent=False,
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=TRUNC_NORMAL_STD)

    @staticmethod
    def _relative_position_index(window_size: int) -> torch.Tensor:
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )                                                        # (2, ws, ws)
        coords = coords.flatten(1)                               # (2, N)
        relative = coords[:, :, None] - coords[:, None, :]       # (2, N, N)
        relative = relative.permute(1, 2, 0).contiguous()        # (N, N, 2)
        relative[..., 0] += window_size - 1
        relative[..., 1] += window_size - 1
        relative[..., 0] *= 2 * window_size - 1
        return relative.sum(-1)                                  # (N, N)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (num_windows*B, N, C)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)           # each (B_, heads, N, head_dim)

        attn = (q * self.scale) @ k.transpose(-2, -1)            # (B_, heads, N, N)

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, self.num_heads).permute(2, 0, 1)  # (heads, N, N)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(B_ // num_windows, num_windows, self.num_heads, N, N)
            attn = attn + mask.to(attn.dtype).unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)       # (B_, N, C)
        return self.proj(out)


class SwinTransformerBlock(nn.Module):
    """Pre-LN window attention with an optional cyclic shift, then an MLP."""

    def __init__(
        self,
        dim:         int,
        num_heads:   int,
        window_size: int = DEFAULT_WINDOW_SIZE,
        shift_size:  int = 0,
        mlp_ratio:   float = DEFAULT_MLP_RATIO,
        qkv_bias:    bool = True,
        attn_drop:   float = 0.0,
        drop_path:   float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, H*W, C) -> (B, H*W, C)
        B, L, C = x.shape
        shortcut = x

        x = self.norm1(x).view(B, H, W, C)

        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        if pad_b or pad_r:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        padded_h, padded_w = H + pad_b, W + pad_r

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        mask = build_attention_mask(
            padded_h, padded_w, H, W, self.window_size, self.shift_size, x.device
        )

        windows = window_partition(x, self.window_size)          # (nW*B, ws**2, C)
        windows = self.attn(windows, mask)
        x = window_reverse(windows, self.window_size, padded_h, padded_w)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        if pad_b or pad_r:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x.view(B, L, C))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class SwinStage(nn.Module):
    """Consecutive Swin blocks alternating between regular and shifted windows."""

    def __init__(
        self,
        dim:         int,
        depth:       int,
        num_heads:   int,
        window_size: int,
        mlp_ratio:   float,
        qkv_bias:    bool,
        attn_drop:   float,
        drop_path:   list[float],
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if index % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    drop_path=drop_path[index],
                )
                for index in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, H, W)
        return x


class PatchEmbed(nn.Module):
    """Split the input into patches and project them to tokens.

    The projection carries no bias: the LayerNorm that follows removes the per-token
    mean, so a pre-norm bias barely shifts the output yet collects the gradient summed
    over every token, which reached 2.6e6 here and overflowed fp16 under autocast.
    The LayerNorm's own bias supplies the shift.
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, patch_size, stride=patch_size, bias=False
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        # (B, C, H, W) -> (B, H/p * W/p, embed_dim)
        x = self.proj(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W


class PatchMerging(nn.Module):
    """Halve the resolution and double the channels."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, MERGE_FACTOR * dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # (B, H*W, C) -> (B, H/2 * W/2, 2C)
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        quadrants = [
            x[:, 0::2, 0::2, :],
            x[:, 1::2, 0::2, :],
            x[:, 0::2, 1::2, :],
            x[:, 1::2, 1::2, :],
        ]
        x = torch.cat(quadrants, dim=-1).view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


class SwinEncoder(nn.Module):
    """Hierarchical Swin encoder that emits one feature map per stage.

    Embeds patch_size x patch_size patches, then runs one stage per resolution with a
    PatchMerging step between consecutive stages, so the spatial dimensions shrink by
    patch_size * 2**(num_stages-1) overall while the channel width doubles per stage.

    forward pads H and W up to that stride, so any input size works; the caller crops
    the decoded output back to the original size.
    """

    def __init__(
        self,
        in_channels:    int = 15,
        embed_dim:      int = DEFAULT_EMBED_DIM,
        depths:         list[int] = None,
        num_heads:      list[int] = None,
        window_size:    int = DEFAULT_WINDOW_SIZE,
        mlp_ratio:      float = DEFAULT_MLP_RATIO,
        qkv_bias:       bool = True,
        drop_path_rate: float = DEFAULT_DROP_PATH_RATE,
        attn_drop:      float = 0.0,
        patch_size:     int = DEFAULT_PATCH_SIZE,
    ) -> None:
        super().__init__()
        depths = list(DEFAULT_DEPTHS if depths is None else depths)
        num_heads = list(DEFAULT_NUM_HEADS if num_heads is None else num_heads)
        if len(depths) != len(num_heads):
            raise ValueError(
                f"depths and num_heads must have equal length, got "
                f"{len(depths)} and {len(num_heads)}"
            )

        num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.stride = patch_size * MERGE_FACTOR ** (num_stages - 1)
        self.stage_channels = [
            embed_dim * MERGE_FACTOR ** stage for stage in range(num_stages)
        ]

        drop_rates = torch.linspace(0.0, drop_path_rate, sum(depths)).tolist()

        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)

        self.stages = nn.ModuleList()
        offset = 0
        for stage in range(num_stages):
            self.stages.append(
                SwinStage(
                    dim=self.stage_channels[stage],
                    depth=depths[stage],
                    num_heads=num_heads[stage],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    drop_path=drop_rates[offset: offset + depths[stage]],
                )
            )
            offset += depths[stage]
        self.downsamples = nn.ModuleList(
            [PatchMerging(self.stage_channels[stage]) for stage in range(num_stages - 1)]
        )

        self.apply(self._init_weights)

    def _get_stage_channels(self) -> list[int]:
        """Return the channel width of each encoder stage, shallowest first."""
        return list(self.stage_channels)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=TRUNC_NORMAL_STD)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # x: (B, in_channels, H, W) -> bottleneck (B, C_last, H/stride, W/stride)
        # and one (B, C_stage, H_stage, W_stage) map per stage.
        B = x.shape[0]
        pad_b = (self.stride - x.shape[2] % self.stride) % self.stride
        pad_r = (self.stride - x.shape[3] % self.stride) % self.stride
        if pad_b or pad_r:
            x = F.pad(x, (0, pad_r, 0, pad_b), mode="replicate")

        tokens, height, width = self.patch_embed(x)          # (B, H/p * W/p, embed_dim)

        skips: list[torch.Tensor] = []
        for stage, encoder in enumerate(self.stages):
            tokens = encoder(tokens, height, width)
            # (B, L, C) -> (B, C, H, W) so the conv decoder can consume it
            skips.append(tokens.transpose(1, 2).reshape(B, -1, height, width))
            if stage < len(self.downsamples):
                tokens = self.downsamples[stage](tokens, height, width)
                height, width = height // MERGE_FACTOR, width // MERGE_FACTOR

        return skips[-1], skips


class SwinTransformer(nn.Module):
    """Swin encoder paired with a UNet decoder.

    Window attention handles encoding; decoding is bilinear upsampling, skip
    concatenation and convolutions. Input and output share the spatial size.
    """

    def __init__(
        self,
        in_channels:    int = 15,
        out_channels:   int = 15,
        embed_dim:      int = DEFAULT_EMBED_DIM,
        depths:         list[int] = None,
        num_heads:      list[int] = None,
        window_size:    int = DEFAULT_WINDOW_SIZE,
        mlp_ratio:      float = DEFAULT_MLP_RATIO,
        qkv_bias:       bool = True,
        drop_path_rate: float = DEFAULT_DROP_PATH_RATE,
        attn_drop:      float = 0.0,
        patch_size:     int = DEFAULT_PATCH_SIZE,
        decoder_blocks: list[int] = None,
        final_blocks:   int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        self.encoder = SwinEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            attn_drop=attn_drop,
            patch_size=patch_size,
        )
        self.decoder = SwinUNetDecoder(
            encoder_channels=self.encoder._get_stage_channels(),
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            decoder_blocks=decoder_blocks,
            final_blocks=final_blocks,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H, W) -> (B, out_channels, H, W)
        bottleneck, skips = self.encoder(x)
        # The decoder also reads x directly: it holds the full-resolution detail the
        # encoder discarded at patch embedding, and it crops the encoder padding.
        return self.decoder(bottleneck, skips, x)
