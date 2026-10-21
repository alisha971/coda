"""p-values from donor-null distributions, and the Holm correction.

**Registered p-value.** p = the proportion of null draws at or above the
observed statistic, computed exactly as written. With B draws its
resolution is 1/B; p = 0 means "below 1/B" and is reported that way, following
the caution of Phipson & Smyth (2010) that a Monte-Carlo permutation p-value
should not be read as exactly zero.

**Holm** (1979) step-down correction is applied to the family H2-H4 only. The
primary test (H1) is a single declared test and is not corrected. The neural
arm's p-values never enter the family.
"""

from __future__ import annotations

import numpy as np


def donor_p_value(observed: float, null: np.ndarray) -> dict:
    null = np.asarray(null, dtype=np.float64)
    null = null[np.isfinite(null)]
    if null.size == 0:
        raise ValueError("empty null distribution")
    p = float((null >= observed).mean())
    return {"p": p, "n_draws": int(null.size), "resolution": 1.0 / null.size,
            "reported": f"< {1.0 / null.size:.4g}" if p == 0 else f"{p:.4g}",
            "observed": float(observed), "null_mean": float(null.mean()),
            "null_q05": float(np.quantile(null, 0.05)),
            "null_q95": float(np.quantile(null, 0.95)),
            "excess_over_null_mean": float(observed - null.mean())}


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-adjusted p-values and rejections for a family of hypotheses."""
    names = list(p_values)
    p = np.array([p_values[n] for n in names], dtype=np.float64)
    order = np.argsort(p, kind="stable")
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return {n: {"p": float(p[i]), "p_holm": float(adj[i]), "reject": bool(adj[i] <= alpha)}
            for i, n in enumerate(names)}


__all__ = ["donor_p_value", "holm"]
