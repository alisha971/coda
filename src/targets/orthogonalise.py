"""Within-block orthogonalisation of the target.

Dimensions inside a block are correlated (mean intensity and intensity range,
for example), and block-level partial R² sums squared errors over a block's
dimensions. Without decorrelation, a block's statistic would count shared
variance more than once.

**Method: ZCA-cor whitening per block** (Kessy, Lewin & Strimmer 2018). For a
block with covariance Σ = V^{1/2} P V^{1/2} (V the variances, P the
correlation matrix), the transform is W = P^{-1/2} V^{-1/2}. Among all
whitening transforms it keeps each output variable maximally correlated with
its own input, so every output keeps the name of the dimension it came from.
Single-dimension blocks are centred and scaled. Gram-Schmidt was rejected
because its result depends on the order of the dimensions.

The transform is fitted on training targets only and applied unchanged to
validation and test.
"""

from __future__ import annotations

import numpy as np

from src.config import TARGET_BLOCKS, TARGET_DIMENSIONS
from src.features.normstats import FittedStats


def _inv_sqrt_psd(m: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    w, v = np.linalg.eigh(m)
    if (w <= eps).any():
        raise np.linalg.LinAlgError("block correlation matrix is singular")
    return (v / np.sqrt(w)) @ v.T


class BlockOrthogonaliser:
    """ZCA-cor whitening applied block by block."""

    def __init__(self, blocks: dict[str, tuple[str, ...]] = TARGET_BLOCKS,
                 dims: tuple[str, ...] = TARGET_DIMENSIONS):
        self.blocks = blocks
        self.dims = list(dims)
        self.index = {b: [self.dims.index(d) for d in ds] for b, ds in blocks.items()}
        self.fitted: FittedStats | None = None

    def fit(self, Y: np.ndarray, fit_ids, split_hash: str) -> BlockOrthogonaliser:
        if not np.isfinite(Y).all():
            raise ValueError("orthogonalisation needs complete training targets")
        params = {}
        for b, idx in self.index.items():
            yb = Y[:, idx]
            mu = yb.mean(axis=0)
            sd = yb.std(axis=0, ddof=1)
            if (sd <= 0).any():
                raise ValueError(f"block {b} has a constant dimension")
            corr = np.corrcoef(yb, rowvar=False) if len(idx) > 1 else np.ones((1, 1))
            # W = P^{-1/2} V^{-1/2}; whitened column vector z = W (y - mu).
            W = _inv_sqrt_psd(np.atleast_2d(corr)) @ np.diag(1 / sd)
            params[b] = {"mean": mu, "W": W}
        self.fitted = FittedStats("block_orthogonaliser", params,
                                  [str(i) for i in fit_ids], split_hash)
        return self

    def transform(self, Y: np.ndarray) -> np.ndarray:
        if self.fitted is None:
            raise RuntimeError("fit the orthogonaliser on training targets first")
        out = np.empty_like(Y, dtype=np.float64)
        for b, idx in self.index.items():
            p = self.fitted.params[b]
            out[:, idx] = (Y[:, idx] - p["mean"]) @ p["W"].T
        return out


__all__ = ["BlockOrthogonaliser"]
