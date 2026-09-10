import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_block import ConvBlock

# Kaiming's fan_out/ReLU gain suits hidden layers, not the final projection: it starts
# the output far from the target, and lambda_recon amplifies the resulting gradient to
# the edge of fp16 range. A small init starts the residual near zero.
OUTPUT_PROJ_INIT_STD = 0.01

FULL_RES_DIVISOR = 2


class SwinUNetDecoder(nn.Module):
    """UNet decoder for hierarchical encoder features, with a full-resolution path.

    Each level upsamples to the next skip's resolution, concatenates that skip, fuses
    the stacked channels back to the skip width with a 3x3 convolution, then refines
    with a residual ConvBlock. ConvBlock preserves channel count, so the fuse
    convolution carries the reduction.

    The encoder's finest feature map is H/patch_size, so a decoder built only from
    those features is band-limited: upsampling cannot invent detail the encoder never
    saw, and event voxel grids are almost entirely per-pixel high frequency. Two paths
    carry that detail instead:

    - `stem` convolves the input at full resolution and joins the last decoder level,
      so per-pixel structure reaches the output.
    - `output_proj` predicts a residual added to the input when the channel counts
      match, so the model refines the input rather than regenerating it.
    """

    def __init__(
        self,
        encoder_channels: list[int],
        in_channels:      int = 15,
        out_channels:     int = 15,
        patch_size:       int = 4,
        decoder_blocks:   list[int] = None,
        final_blocks:     int = 1,
        bias:             bool = False,
    ) -> None:
        super().__init__()
        if len(encoder_channels) < 2:
            raise ValueError(
                f"Need at least two encoder stages, got {len(encoder_channels)}"
            )

        self.encoder_channels = list(encoder_channels)
        self.patch_size = patch_size
        self.residual = in_channels == out_channels

        skip_channels = self.encoder_channels[:-1]          # [C1, C2, C3]
        deep_channels = self.encoder_channels[1:]           # [C2, C3, C4]
        num_levels = len(skip_channels)

        # ConvBlocks per level, deepest first. One each unless asked for more.
        if decoder_blocks is None:
            decoder_blocks = [1] * num_levels
        decoder_blocks = list(decoder_blocks)
        if len(decoder_blocks) != num_levels:
            raise ValueError(
                f"decoder_blocks needs one entry per decoder level: expected "
                f"{num_levels}, got {len(decoder_blocks)}"
            )
        if min(decoder_blocks) < 1 or final_blocks < 1:
            raise ValueError("Block counts must be at least 1")
        self.decoder_blocks = decoder_blocks
        self.final_blocks = final_blocks

        # One level per skip, deepest first: C4+C3 -> C3, C3+C2 -> C2, C2+C1 -> C1.
        self.fuse = nn.ModuleList(
            [
                nn.Conv2d(deep + skip, skip, 3, padding=1, bias=bias)
                for skip, deep in zip(reversed(skip_channels), reversed(deep_channels))
            ]
        )
        self.process = nn.ModuleList(
            [
                nn.Sequential(*[ConvBlock(skip, bias=bias) for _ in range(count)])
                for skip, count in zip(reversed(skip_channels), decoder_blocks)
            ]
        )

        # Full-resolution head. Kept narrower than C1 — it runs at H x W.
        full_res_dim = max(self.encoder_channels[0] // FULL_RES_DIVISOR, out_channels)
        self.stem = nn.Conv2d(in_channels, full_res_dim, 3, padding=1, bias=bias)
        self.full_res_fuse = nn.Conv2d(
            self.encoder_channels[0] + full_res_dim, full_res_dim, 3, padding=1, bias=bias
        )
        self.refine = nn.Sequential(
            *[ConvBlock(full_res_dim, bias=bias) for _ in range(final_blocks)]
        )
        self.output_proj = nn.Conv2d(
            full_res_dim, out_channels, 3, padding=1, bias=bias
        )

        self._init_conv_weights()

    def _init_conv_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.output_proj.weight, std=OUTPUT_PROJ_INIT_STD)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips:      list[torch.Tensor],
        inp:        torch.Tensor,
    ) -> torch.Tensor:
        # bottleneck: (B, C4, H/32, W/32); skips: [(B,C1,H/4,W/4) ... (B,C4,H/32,W/32)]
        # at the encoder's padded size; inp: (B, in_channels, H, W) unpadded.
        # The last skip is the bottleneck itself, so only the shallower ones fuse in.
        fusible = skips[: len(self.fuse)]

        x = bottleneck
        for level, (fuse, process) in enumerate(zip(self.fuse, self.process)):
            skip = fusible[-(level + 1)]
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
            x = torch.cat([x, skip], dim=1)                 # (B, C_deep + C_skip, H, W)
            x = process(fuse(x))                            # (B, C_skip, H, W)

        x = F.interpolate(
            x, scale_factor=self.patch_size, mode="bilinear", align_corners=False
        )
        x = x[:, :, : inp.shape[2], : inp.shape[3]]         # drop the encoder padding

        x = torch.cat([x, self.stem(inp)], dim=1)           # (B, C1 + full_res, H, W)
        x = self.refine(self.full_res_fuse(x))              # (B, full_res, H, W)
        out = self.output_proj(x)                           # (B, out_channels, H, W)
        return out + inp if self.residual else out
