"""The partner acoustic encoder.

Learned positional encoding; three pre-norm transformer blocks (d = 192, 4 heads,
FFN 384); a key-padding mask from the true frame lengths, so padded frames
never contribute; attention pooling with a learned query, giving a_i ∈ R^192.

Sequences hold at most 400 frames (the final 8 s at 50 Hz), so the
positional table has 400 rows and longer inputs are rejected rather than
silently truncated.
"""

from __future__ import annotations

import torch
from torch import nn


def padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """True at padded positions (the convention of ``nn.MultiheadAttention``)."""
    idx = torch.arange(max_len, device=lengths.device)[None, :]
    return idx >= lengths[:, None]


class AttentionPool(nn.Module):
    """A learned query attends over the (masked) sequence."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query.expand(x.shape[0], -1, -1)
        out, w = self.attn(q, x, x, key_padding_mask=mask, need_weights=True)
        return out[:, 0], w[:, 0]


class TemporalTransformer(nn.Module):
    def __init__(self, d_model: int = 192, n_blocks: int = 3, n_heads: int = 4,
                 ffn_dim: int = 384, max_frames: int = 400, dropout: float = 0.1):
        super().__init__()
        self.max_frames = max_frames
        self.pos = nn.Parameter(torch.randn(max_frames, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ffn_dim, dropout=dropout,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, n_blocks, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.pool = AttentionPool(d_model, n_heads, dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """x: (batch, time, d_model); lengths: (batch,) true frame counts >= 1.

        Returns the encoded frames, the pooled vector a_i, and the padding mask.
        """
        T = x.shape[1]
        if T > self.max_frames:
            raise ValueError(f"{T} frames exceed the {self.max_frames}-frame cap")
        if (lengths < 1).any() or (lengths > T).any():
            raise ValueError("every sequence needs a true length between 1 and its padded size")
        mask = padding_mask(lengths, T)
        h = self.blocks(x + self.pos[:T][None], src_key_padding_mask=mask)
        h = self.norm(h)
        pooled, _ = self.pool(h, mask)
        return h, pooled, mask


__all__ = ["padding_mask", "AttentionPool", "TemporalTransformer"]
