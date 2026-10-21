"""The positive control: semi-synthetic injection (J3).

    Y*_i = f̂(Z_i) + β · g(A_i) + ε_i

- **f̂** — the content-only (``Z_content``) ridge fitted on training pairs,
  applied to every pair.
- **g** — the injected coupling: the identity on matched dimensions (partner
  dimension j drives reply dimension j), or the cross-dimension variant, in
  which each reply dimension is driven by one partner dimension from a
  different block (the fixed mapping in ``Settings.injection``).
- **ε** — real out-of-fold residual vectors of the content-only model. Whole
  8-dimensional vectors are drawn, so the covariance between dimensions is kept,
  and they are drawn **by session block**: every session receives residuals
  from one randomly chosen training session. That keeps between-session
  clustering in the noise while detaching every residual from the pair whose A
  it is now combined with, so at β = 0 the null of no partner coupling holds by
  construction.
- **β** — set per block so that the injected term's expected partial R² equals
  the target value t: β² = t · Var(ε_b) / ((1 − t) · Var(g̃_b)), with g̃ the
  injected term residualised on Z over training rows (sums over the block's
  dimensions). The grid spans partial R² 0.002-0.05.

**Where the injection happens.** In the standardised target space, after
per-speaker standardisation, because f̂ and ε are defined there. The steps after
standardisation (orthogonalisation fit, nested CV, donor nulls, correction) all
run on Y*. Re-standardising Y* per speaker would be an affine near-identity and
is not repeated. This is recorded as a provisional resolution of the requirement that "the
entire pipeline runs on Y*".

Many replicates are generated at once and stacked as target columns, so one
pass of each donor draw serves all of them (ridge solves are column-wise).
"""

from __future__ import annotations

import numpy as np

from src.config import TARGET_BLOCKS, TARGET_DIMENSIONS


def coupling_matrix(kind: str, cross_map: dict[str, str] | None = None) -> np.ndarray:
    """G such that g(A) = A @ G. Identity, or the documented cross-dimension map."""
    dims = list(TARGET_DIMENSIONS)
    m = len(dims)
    if kind == "identity_matched_dimensions":
        return np.eye(m)
    if kind == "cross_dimension_variant":
        if cross_map is None:
            raise ValueError("the cross-dimension variant needs its mapping")
        block_of = {d: b for b, ds in TARGET_BLOCKS.items() for d in ds}
        G = np.zeros((m, m))
        for reply_dim, partner_dim in cross_map.items():
            if block_of[reply_dim] == block_of[partner_dim]:
                raise ValueError(f"cross map keeps {reply_dim} inside its own block")
            G[dims.index(partner_dim), dims.index(reply_dim)] = 1.0
        if sorted(G.sum(axis=0)) != [1.0] * m or sorted(G.sum(axis=1)) != [1.0] * m:
            raise ValueError("the cross map must be a permutation of the eight dimensions")
        return G
    raise ValueError(f"unknown coupling {kind!r}")


def residualise_on_z(X: np.ndarray, Z: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """X minus its least-squares projection on [1, Z], fitted on ``rows``."""
    Zs = np.column_stack([np.ones(len(Z)), (Z - Z[rows].mean(0)) /
                          np.where(Z[rows].std(0) > 0, Z[rows].std(0), 1)])
    coef, *_ = np.linalg.lstsq(Zs[rows], X[rows], rcond=None)
    return X - Zs @ coef


def betas_for_target(target_pr2: float, g_tilde: np.ndarray, eps_pool: np.ndarray,
                     blocks: list[np.ndarray], rows: np.ndarray) -> np.ndarray:
    """β per block giving expected block partial R² = target (column-wise vector)."""
    m = g_tilde.shape[1]
    beta = np.zeros(m)
    if target_pr2 <= 0:
        return beta
    if target_pr2 >= 1:
        raise ValueError("target partial R² must be below 1")
    for b in blocks:
        vg = g_tilde[rows][:, b].var(axis=0).sum()
        ve = eps_pool[:, b].var(axis=0).sum()
        beta[b] = np.sqrt(target_pr2 * ve / ((1 - target_pr2) * vg))
    return beta


def session_block_residuals(eps_pool: np.ndarray, pool_groups: np.ndarray,
                            groups: np.ndarray, rng: np.random.Generator,
                            ordered: bool = False) -> np.ndarray:
    """ε for every row: each session draws its rows from one random pool session.

    ``ordered=False`` (``session_block``) draws rows at random with
    replacement inside the pool session. It keeps the covariance across
    dimensions and the session-level component, but destroys serial
    dependence between consecutive turns.

    ``ordered=True`` (``session_block_ordered``) copies a contiguous run
    of the pool session's rows, in their original order, starting at a random
    offset and wrapping round at the end. It keeps the covariance across
    dimensions and the serial dependence, apart from one join at the wrap.
    Rows must be in conversation order within each session; pairs are built
    that way.
    """
    pool_sessions = np.unique(pool_groups)
    rows_of = {s: np.flatnonzero(pool_groups == s) for s in pool_sessions}
    out = np.empty((len(groups), eps_pool.shape[1]))
    for s in np.unique(groups):
        idx = np.flatnonzero(groups == s)
        src = rows_of[pool_sessions[rng.integers(len(pool_sessions))]]
        if ordered:
            take = src[(rng.integers(len(src)) + np.arange(len(idx))) % len(src)]
        else:
            take = src[rng.integers(0, len(src), len(idx))]
        out[idx] = eps_pool[take]
    return out


def injected_targets(fhat: np.ndarray, A: np.ndarray, Z: np.ndarray, eps_pool: np.ndarray,
                     pool_groups: np.ndarray, groups: np.ndarray, train_rows: np.ndarray,
                     target_pr2: float, G: np.ndarray, blocks: list[np.ndarray],
                     n_replicates: int, seed: int, ordered: bool = False,
                     replicate_ids=None) -> tuple[np.ndarray, np.ndarray]:
    """Stacked Y* (n × n_replicates·m) and the β used per column of one replicate.

    Replicate ``r`` draws its residuals from its own generator, seeded by
    ``(seed, r)``. A level can therefore be computed in chunks of replicates
    (``replicate_ids``) and give exactly the replicates an unchunked run gives.
    """
    gA = A @ G
    g_tilde = residualise_on_z(gA, Z, train_rows)
    beta = betas_for_target(target_pr2, g_tilde, eps_pool, blocks, train_rows)
    ids = range(n_replicates) if replicate_ids is None else replicate_ids
    reps = []
    for r in ids:
        rng = np.random.default_rng([seed, int(r)])
        eps = session_block_residuals(eps_pool, pool_groups, groups, rng, ordered)
        reps.append(fhat + gA * beta + eps)
    return np.hstack(reps), beta


def rejection_rate(p_values: np.ndarray, alpha: float) -> dict:
    """Share of replicates rejecting, with its binomial Monte-Carlo interval."""
    from scipy.stats import binomtest

    p = np.asarray(p_values)
    k, n = int((p <= alpha).sum()), int(p.size)
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return {"rate": k / n if n else float("nan"), "rejections": k, "replicates": n,
            "ci95": [float(ci.low), float(ci.high)]}


def mde_from_power(targets: list[float], power: list[float], level: float) -> float | None:
    """Smallest injected partial R² at which power reaches ``level`` (linear
    interpolation between grid points); None if it never does (Bloom 1995)."""
    t, pw = np.asarray(targets, float), np.asarray(power, float)
    order = np.argsort(t)
    t, pw = t[order], pw[order]
    above = np.flatnonzero(pw >= level)
    if above.size == 0:
        return None
    i = above[0]
    if i == 0:
        return float(t[0])
    t0, t1, p0, p1 = t[i - 1], t[i], pw[i - 1], pw[i]
    return float(t0 + (level - p0) * (t1 - t0) / (p1 - p0)) if p1 > p0 else float(t1)


__all__ = ["coupling_matrix", "residualise_on_z", "betas_for_target", "session_block_residuals",
           "injected_targets", "rejection_rate", "mde_from_power"]
