"""The delivery head and the block-aware objective.

**Head.** A 2-layer MLP to the eight target means, plus eight log-variances in
the optional heteroscedastic mode (the output layer always has 16 units so the
parameter count does not depend on the mode; in homoscedastic mode the
log-variance half is ignored).

**Objective.** L = Σ_b w_b · (1/|b|) Σ_{j∈b} loss_j, with w_b = 1/4 for each of
the four blocks, so Timing (three dimensions) cannot dominate Pitch boundary
(one) through count. loss_j is squared error on the standardised, orthogonalised
target; in heteroscedastic mode it is the Gaussian negative log-likelihood. The
heteroscedastic mode is entered only after a block-aware MSE warm start (the
guard against variance collapse) and is kept only if it improves
validation log-likelihood.
"""

from __future__ import annotations

import torch
from torch import nn


class DeliveryHead(nn.Module):
    def __init__(self, in_dim: int = 384, hidden: int = 192, n_targets: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.n_targets = n_targets
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 2 * n_targets))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        return out[:, :self.n_targets], out[:, self.n_targets:]


def block_weights(blocks: list[list[int]], n_targets: int) -> torch.Tensor:
    """Per-dimension weights: 1/(n_blocks · |b|), so each block sums to 1/n_blocks."""
    w = torch.zeros(n_targets)
    for b in blocks:
        w[b] = 1.0 / (len(blocks) * len(b))
    if not torch.isclose(w.sum(), torch.tensor(1.0)):
        raise ValueError("block weights must sum to one")
    return w


def block_loss(mean: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor,
               weights: torch.Tensor, heteroscedastic: bool) -> torch.Tensor:
    if heteroscedastic:
        per = 0.5 * (log_var + (y - mean) ** 2 * torch.exp(-log_var))
    else:
        per = (y - mean) ** 2
    return (per.mean(dim=0) * weights.to(per.device)).sum()


__all__ = ["DeliveryHead", "block_weights", "block_loss"]
