"""The conditional variance model.

A second ridge, fitted on the squared out-of-fold residuals of the mean model,
predicts how uncertain each pair's prediction is. It supplies the predictive
intervals (coverage at 68% and 95%) and the variance in the log-likelihood
unit. The registered form is a ridge on squared residuals; its predictions can
fall below zero, so they are floored at a small fraction of the mean squared
residual. How often the floor binds is reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.ridge import RidgeEngine, ZCache


@dataclass
class VarianceFit:
    var_eval: np.ndarray        # (n_eval, m)
    floor: np.ndarray           # (m,)
    floored_share: float
    lambdas: np.ndarray


def fit_variance(zcache: ZCache, oof_residuals: np.ndarray, A: np.ndarray | None,
                 lambda_rel_grid, floor_frac: float = 0.01) -> VarianceFit:
    """Ridge on squared OOF residuals, one λ per target column."""
    n_all = zcache.Z.shape[0]
    m = oof_residuals.shape[1]
    sq = np.full((n_all, m), np.nan)
    sq[zcache.fit_idx] = oof_residuals ** 2
    sq[zcache.eval_idx] = 0.0            # placeholder; evaluation targets are never read
    eng = RidgeEngine(zcache, sq, [np.array([j]) for j in range(m)], lambda_rel_grid)
    res = eng.run(A)
    floor = floor_frac * (oof_residuals ** 2).mean(axis=0)
    var = np.maximum(res.pred_eval, floor)
    return VarianceFit(var, floor, float((res.pred_eval < floor).mean()), res.lambdas)


__all__ = ["VarianceFit", "fit_variance"]
