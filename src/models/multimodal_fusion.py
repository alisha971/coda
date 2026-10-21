"""Cross-modal fusion: one cross-attention block.

The reply-side context c_i and the partner-text representation p_i are encoded
separately and only then combined (modality separation is explicit; nothing is
concatenated as raw input). The query [c_i ; p_i] (96 + 96 = 192) attends over
the masked partner acoustic frames: "given what is about to be said and by
whom, which parts of how the partner spoke matter". The pooled partner vector
a_i is added back as a residual path, and the output is [c_i ; p_i ; attended].

One block. It generates nothing: no decoder, no tokens, no audio.
"""

from __future__ import annotations

import torch
from torch import nn


class CrossModalFusion(nn.Module):
    def __init__(self, d_model: int = 192, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, c: torch.Tensor, p: torch.Tensor, frames: torch.Tensor,
                mask: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(torch.cat([c, p], dim=-1))[:, None, :]
        kv = self.norm_kv(frames)
        att, _ = self.attn(q, kv, kv, key_padding_mask=mask, need_weights=False)
        attended = att[:, 0] + pooled
        return torch.cat([c, p, attended], dim=-1)


__all__ = ["CrossModalFusion"]
