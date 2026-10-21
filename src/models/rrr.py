"""Reduced-rank ridge for the coupling map.

The partner → reply coefficient block β_A (8 partner dimensions × 8 reply
dimensions) is constrained to rank r, with Z profiled out and penalised as in
the ridge (see :meth:`src.models.ridge.RidgeEngine.rank_constrained`). The
factorisation gives, for each latent direction k:

- **partner loadings**: how the partner dimensions combine (column k of β̂_A Q);
- **reply loadings**: which reply dimensions the direction moves (column k of Q);
- **strength** d_k: the singular value of S^{1/2} β̂_A, so d_k² is the share of
  the fitted coupling the direction carries.

λ and rank are chosen jointly by the same grouped inner folds as the ridge,
minimising out-of-fold SSE weighted equally per block (each block's SSE divided
by its number of dimensions), the same weighting the confirmatory statistic
uses. Reference: Mukherjee & Zhu (2011), reduced-rank ridge regression; the
reduced-rank model itself goes back to Izenman (1975).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.ridge import RidgeEngine


@dataclass
class RRRResult:
    lam: float
    lam_rel: float
    rank: int
    at_grid_edge: bool
    cv: np.ndarray                    # (n_lambda, n_rank) weighted OOF SSE
    partner_loadings: np.ndarray      # (q, rank)
    reply_loadings: np.ndarray        # (m, rank)
    strengths: np.ndarray             # all singular values d (descending)
    shares: np.ndarray                # d² / Σ d²
    beta_A: np.ndarray                # unconstrained ridge β̂_A at the chosen λ (q × m)
    pred_eval: np.ndarray


def block_weights(blocks: list[np.ndarray], m: int) -> np.ndarray:
    w = np.zeros(m)
    for b in blocks:
        w[b] = 1.0 / len(b)
    return w


def fit_rrr(engine: RidgeEngine, A: np.ndarray, blocks: list[np.ndarray],
            ranks=(1, 2, 3, 4, 5, 6, 7, 8), fixed: tuple[float, int] | None = None) -> RRRResult:
    """Reduced-rank ridge of Y on [Z A]; λ and rank by grouped inner CV.

    ``fixed=(lambda_rel, rank)`` skips selection, for bootstrap and fold refits
    whose λ and rank must stay those of the reference fit.
    """
    A = np.asarray(A, dtype=np.float64)
    m = engine.Y.shape[1]
    ranks = [r for r in ranks if r <= min(A.shape[1], m)]
    w = block_weights(blocks, m)
    grid = engine.grid_rel if fixed is None else np.array([fixed[0]])
    ranks = ranks if fixed is None else [fixed[1]]
    cv = np.zeros((len(grid), len(ranks)))
    if fixed is None:
        for fz in engine.folds[:-1]:
            ap = engine._a_pieces(fz, A)
            for li, g in enumerate(grid):
                preds, _ = engine.rank_constrained(fz, ap, g * len(fz.fit), ranks)
                for ri, r in enumerate(ranks):
                    cv[li, ri] += (((fz.Ye - preds[r]) ** 2).sum(axis=0) * w).sum()
    li, ri = np.unravel_index(int(np.argmin(cv)), cv.shape)
    lam_rel, rank = float(grid[li]), int(ranks[ri])
    final = engine.folds[-1]
    ap = engine._a_pieces(final, A)
    lam = lam_rel * len(final.fit)
    preds, f = engine.rank_constrained(final, ap, lam, [rank])
    Q = f["Q"][:, :rank]
    d = f["d"]
    shares = d ** 2 / (d ** 2).sum() if (d ** 2).sum() > 0 else np.zeros_like(d)
    edge = fixed is None and (li in (0, len(grid) - 1))
    return RRRResult(lam=lam, lam_rel=lam_rel, rank=rank, at_grid_edge=bool(edge), cv=cv,
                     partner_loadings=f["beta_A"] @ Q, reply_loadings=Q, strengths=d,
                     shares=shares, beta_A=f["beta_A"], pred_eval=preds[rank])


def leading_strength(engine: RidgeEngine, A: np.ndarray, lam_rel: float) -> float:
    """d₁ of the final fit at a fixed λ: the statistic for the H2 donor null."""
    final = engine.folds[-1]
    ap = engine._a_pieces(final, np.asarray(A, dtype=np.float64))
    _, f = engine.rank_constrained(final, ap, lam_rel * len(final.fit), [1])
    return float(f["d"][0])


__all__ = ["RRRResult", "fit_rrr", "leading_strength", "block_weights"]
