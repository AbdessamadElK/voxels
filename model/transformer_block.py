import torch
import torch.nn as nn

FFN_EXPANSION = 2.66


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer encoder block: MHSA + pointwise Conv FFN, both with residuals."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float = FFN_EXPANSION,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, bias=bias, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * ffn_expansion_factor)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, bias=bias),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        if self.attn.in_proj_bias is not None:
            nn.init.zeros_(self.attn.in_proj_bias)
        nn.init.xavier_uniform_(self.attn.out_proj.weight)
        if self.attn.out_proj.bias is not None:
            nn.init.zeros_(self.attn.out_proj.bias)
        for m in self.ffn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)                    # (B, C, H, W) -> (B, H*W, C)

        normed = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens = tokens + attn_out                               # (B, H*W, C)

        normed2 = self.norm2(tokens)                             # (B, H*W, C)
        feat = normed2.transpose(1, 2).reshape(B, C, H, W)      # (B, C, H, W)
        residual = tokens.transpose(1, 2).reshape(B, C, H, W)   # (B, C, H, W)
        return residual + self.ffn(feat)                         # (B, C, H, W)
