"""The learned mixture over five frozen SSL sources.

    H_mix[t] = Σ_l α_l · Proj_l(H_l[t]),   α = softmax(w),   w learned

Five sources (WavLM layers 3, 6, 9, 12 and HuBERT's final layer), each already
reduced by PCA to 128 dimensions in the cache, are projected to d = 192 and
combined with softmax weights. The weights are a parameter trained on training
data like any other (layer-mixture row): no layer is chosen on validation
or test performance. α is logged per run and reported (Figure 8) as a modelling
statement, never as a finding about which layer "wins".

**Ablation 4** (mixer frozen to a single source) sets the logits so that
α is one-hot on the chosen source and stops their gradient. Every
parameter still exists with the same shape, so capacity is unchanged.
"""

from __future__ import annotations

import torch
from torch import nn

SOURCES = ("wavlm_l3", "wavlm_l6", "wavlm_l9", "wavlm_l12", "hubert_final")


class SSLLayerMixer(nn.Module):
    def __init__(self, in_dim: int = 128, d_model: int = 192, n_sources: int = len(SOURCES)):
        super().__init__()
        self.proj = nn.ModuleList(nn.Linear(in_dim, d_model) for _ in range(n_sources))
        self.logits = nn.Parameter(torch.zeros(n_sources))      # uniform α at initialisation
        self.n_sources = n_sources

    def freeze_to(self, source: int) -> None:
        """Ablation 4: α fixed one-hot on ``source``; shapes unchanged."""
        with torch.no_grad():
            self.logits.fill_(-1e4)
            self.logits[source] = 0.0
        self.logits.requires_grad_(False)

    def alpha(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (batch, time, n_sources, in_dim) -> (batch, time, d_model)."""
        if frames.shape[2] != self.n_sources:
            raise ValueError(f"expected {self.n_sources} sources, got {frames.shape[2]}")
        a = self.alpha()
        out = 0.0
        for k, proj in enumerate(self.proj):
            out = out + a[k] * proj(frames[:, :, k, :])
        return out


__all__ = ["SOURCES", "SSLLayerMixer"]
