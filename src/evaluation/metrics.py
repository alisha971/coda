"""The two registered units and the coverage check.

- **Block partial R²** (primary unit): for block b, 1 − ΣSSE_full / ΣSSE_reduced,
  sums over the block's (orthogonalised) dimensions and over evaluation pairs.
  It is the share of the reduced model's held-out error that adding the extra
  block removes. Negative values are possible on held-out data and are read only
  against the null, never as "the extra information hurts".
- **Aggregate**: the equal-weight mean of the four block values.
- **Log-likelihood increment** (second unit): mean over pairs of the per-pair
  difference in Gaussian log-likelihood, using each model's predicted
  variances from the conditional variance model.
- **Interval coverage** at 68% and 95% per block: the share of held-out
  values inside mean ± z·σ.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def sse_by_column(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return ((y - pred) ** 2).sum(axis=0)


def block_partial_r2(y: np.ndarray, pred_full: np.ndarray, pred_reduced: np.ndarray,
                     blocks: list[np.ndarray]) -> np.ndarray:
    full, red = sse_by_column(y, pred_full), sse_by_column(y, pred_reduced)
    return np.array([1.0 - full[b].sum() / red[b].sum() for b in blocks])


def block_partial_r2_from_sse(sse_full: np.ndarray, sse_reduced: np.ndarray,
                              blocks: list[np.ndarray]) -> np.ndarray:
    """Block partial R² from per-column SSE (same definition as above)."""
    return np.array([1.0 - sse_full[b].sum() / sse_reduced[b].sum() for b in blocks])


def aggregate(block_values: np.ndarray) -> float:
    """Equal weight per block."""
    return float(np.mean(block_values))


def gaussian_loglik(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Per-pair, per-dimension Gaussian log-likelihood."""
    return -0.5 * (np.log(2 * np.pi * var) + (y - mean) ** 2 / var)


def loglik_increment(y, mean_full, var_full, mean_red, var_red,
                     blocks: list[np.ndarray]) -> np.ndarray:
    """Mean over pairs of the per-block log-likelihood difference (full − reduced)."""
    d = gaussian_loglik(y, mean_full, var_full) - gaussian_loglik(y, mean_red, var_red)
    return np.array([d[:, b].sum(axis=1).mean() for b in blocks])


def interval_coverage(y, mean, var, blocks: list[np.ndarray],
                      levels=(0.68, 0.95)) -> dict[float, np.ndarray]:
    sd = np.sqrt(var)
    out = {}
    for lv in levels:
        z = norm.ppf(0.5 + lv / 2)
        inside = np.abs(y - mean) <= z * sd
        out[lv] = np.array([inside[:, b].mean() for b in blocks])
    return out


def per_pair_errors(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Squared errors per pair and column, stored so the bootstrap can recompute
    partial R² without refitting."""
    return (y - pred) ** 2


__all__ = ["sse_by_column", "block_partial_r2", "block_partial_r2_from_sse", "aggregate",
           "gaussian_loglik", "loglik_increment", "interval_coverage", "per_pair_errors"]
