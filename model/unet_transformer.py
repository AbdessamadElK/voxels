import torch
import torch.nn as nn

from .transformer_block import TransformerBlock

DEFAULT_DIM = 48
FFN_EXPANSION = 2.66


class UNetTransformer(nn.Module):
    """Restormer-skeleton UNet with vanilla transformer blocks (pre-LN MHSA + GELU FFN).

    Encoder compresses (B, in_channels, H, W) through two stride-2 downsamples.
    Decoder mirrors via transposed convolutions with skip connections.
    """

    def __init__(
        self,
        in_channels: int,
        dim: int = DEFAULT_DIM,
        num_blocks: list[int] = None,
        num_heads: list[int] = None,
        ffn_expansion_factor: float = FFN_EXPANSION,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if num_heads is None:
            num_heads = [1, 2, 4, 8]

        kw = dict(ffn_expansion_factor=ffn_expansion_factor, bias=bias)

        # (B, in_channels, H, W) -> (B, dim, H, W)
        self.patch_embed = nn.Conv2d(in_channels, dim, 3, padding=1, bias=bias)

        # Encoder
        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim, num_heads[0], **kw) for _ in range(num_blocks[0])]
        )
        # (B, dim, H, W) -> (B, dim*2, H/2, W/2)
        self.down1 = nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1, bias=bias)

        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim * 2, num_heads[1], **kw) for _ in range(num_blocks[1])]
        )
        # (B, dim*2, H/2, W/2) -> (B, dim*4, H/4, W/4)
        self.down2 = nn.Conv2d(dim * 2, dim * 4, 4, stride=2, padding=1, bias=bias)

        # Bottleneck
        self.latent = nn.Sequential(
            *[TransformerBlock(dim * 4, num_heads[2], **kw) for _ in range(num_blocks[2])]
        )

        # Decoder
        # (B, dim*4, H/4, W/4) -> (B, dim*2, H/2, W/2)
        self.up2 = nn.ConvTranspose2d(dim * 4, dim * 2, 2, stride=2, bias=bias)
        # input after skip concat: dim*2 + dim*2 = dim*4
        self.decoder_level2 = nn.Sequential(
            *[TransformerBlock(dim * 4, num_heads[2], **kw) for _ in range(num_blocks[1])]
        )

        # (B, dim*4, H/2, W/2) -> (B, dim, H, W)
        self.up1 = nn.ConvTranspose2d(dim * 4, dim, 2, stride=2, bias=bias)
        # input after skip concat: dim + dim = dim*2
        self.decoder_level1 = nn.Sequential(
            *[TransformerBlock(dim * 2, num_heads[1], **kw) for _ in range(num_blocks[0])]
        )

        # (B, dim*2, H, W) -> (B, in_channels, H, W)
        self.output_proj = nn.Conv2d(dim * 2, in_channels, 3, padding=1, bias=bias)

        self._init_conv_weights()

    def _init_conv_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H, W)
        inp = self.patch_embed(x)                              # (B, dim, H, W)

        enc1 = self.encoder_level1(inp)                        # (B, dim, H, W)
        enc2 = self.encoder_level2(self.down1(enc1))           # (B, dim*2, H/2, W/2)
        lat = self.latent(self.down2(enc2))                    # (B, dim*4, H/4, W/4)

        dec2 = self.decoder_level2(
            torch.cat([self.up2(lat), enc2], dim=1)            # (B, dim*4, H/2, W/2)
        )
        dec1 = self.decoder_level1(
            torch.cat([self.up1(dec2), enc1], dim=1)           # (B, dim*2, H, W)
        )

        return self.output_proj(dec1)                          # (B, in_channels, H, W)
