import torch
import torch.nn as nn

FFN_EXPANSION = 2.66


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer encoder block: MHSA + GELU FFN, both with residuals."""

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
            nn.Linear(dim, hidden, bias=bias),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=bias),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        if self.attn.in_proj_bias is not None:
            nn.init.zeros_(self.attn.in_proj_bias)
        nn.init.xavier_uniform_(self.attn.out_proj.weight)
        if self.attn.out_proj.bias is not None:
            nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)           # (B, C, H, W) -> (B, H*W, C)

        normed = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens = tokens + attn_out                       # residual

        normed = self.norm2(tokens)
        tokens = tokens + self.ffn(normed)               # residual

        return tokens.transpose(1, 2).reshape(B, C, H, W)  # (B, H*W, C) -> (B, C, H, W)
