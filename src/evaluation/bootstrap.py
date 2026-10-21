"""Paired session-level cluster bootstrap, with its self-test.

The unit of inference is the session. Intervals for held-out statistics
are obtained by resampling **test sessions** with replacement and recomputing
the statistic from the stored per-pair squared errors of both models, so the
full and reduced models are always evaluated on the same resampled pairs
(paired). Models are not refitted: the interval describes uncertainty about the
held-out estimate across sessions, not across training sets.

Resampling whole sessions keeps every within-session dependence intact; a
pair-level bootstrap would treat ~400 dependent pairs per session as
independent and understate uncertainty. Field & Welsh (2007) show the cluster
bootstrap gives consistent variance estimates under both random-effect and
transformation models of clustering; Cameron, Gelbach & Miller (2008) discuss
its behaviour with few clusters. With roughly 50 test sessions, the percentile
interval is used and its coverage is checked by :func:`self_test`.
"""

from __future__ import annotations

import numpy as np


def session_sums(values: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-session column sums of ``values``; returns (sums, session labels)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.zeros((len(uniq), values.shape[1]))
    np.add.at(sums, inv, values)
    return sums, uniq


def resample_counts(n_sessions: int, n_boot: int, seed: int) -> np.ndarray:
    """How many times each session appears in each resample (B × n_sessions)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_sessions, size=(n_boot, n_sessions))
    counts = np.zeros((n_boot, n_sessions))
    np.add.at(counts, (np.repeat(np.arange(n_boot), n_sessions), idx.ravel()), 1)
    return counts


def paired_partial_r2(err_full: np.ndarray, err_red: np.ndarray, groups: np.ndarray,
                      blocks: list[np.ndarray], n_boot: int = 2000, seed: int = 20260912,
                      levels=(0.68, 0.95)) -> dict:
    """Bootstrap distribution and percentile intervals of block and aggregate partial R²."""
    sf, uniq = session_sums(err_full, groups)
    sr, _ = session_sums(err_red, groups)
    counts = resample_counts(len(uniq), n_boot, seed)
    bf, br = counts @ sf, counts @ sr                      # (B × m)
    blk = np.stack([1 - bf[:, b].sum(1) / br[:, b].sum(1) for b in blocks], axis=1)
    agg = blk.mean(axis=1)
    point_blk = np.array([1 - sf[:, b].sum() / sr[:, b].sum() for b in blocks])
    out = {"n_sessions": int(len(uniq)), "n_boot": n_boot, "seed": seed,
           "block_point": point_blk.tolist(), "aggregate_point": float(point_blk.mean()),
           "block_draws": blk, "aggregate_draws": agg, "intervals": {}}
    for lv in levels:
        lo, hi = 0.5 - lv / 2, 0.5 + lv / 2
        out["intervals"][lv] = {
            "block": np.quantile(blk, [lo, hi], axis=0).T.tolist(),
            "aggregate": np.quantile(agg, [lo, hi]).tolist()}
    return out


def _aggregate(f: np.ndarray, r: np.ndarray, blocks: list[np.ndarray]) -> np.ndarray:
    """Aggregate partial R² from error totals; leading axes are kept."""
    return np.mean([1 - f[..., b].sum(-1) / r[..., b].sum(-1) for b in blocks], axis=0)


def interval_methods(err_full: np.ndarray, err_red: np.ndarray, groups: np.ndarray,
                     blocks: list[np.ndarray], n_boot: int = 2000, seed: int = 20260912,
                     level: float = 0.95) -> dict:
    """Percentile, BCa and cluster bootstrap-t intervals of the aggregate partial R².

    All three use the same resamples of whole sessions (models not refitted).
    BCa takes its acceleration from the leave-one-session-out jackknife (Efron
    1987). Bootstrap-t studentizes each resample by its own leave-one-session
    jackknife standard error (one copy of a repeated session removed at a
    time), following the cluster bootstrap-t of Cameron, Gelbach & Miller
    (2008). The method used for reporting is set in ``Settings.bootstrap``; this function serves the
    comparison in ``docs/research/simulations/BOOTSTRAP_COMPARISON_PROTOCOL.md``.
    """
    from scipy.stats import norm

    sf, _ = session_sums(err_full, groups)
    sr, _ = session_sums(err_red, groups)
    n = len(sf)
    F, R = sf.sum(0), sr.sum(0)
    T = float(_aggregate(F, R, blocks))
    counts = resample_counts(n, n_boot, seed)
    Fb, Rb = counts @ sf, counts @ sr
    Tb = _aggregate(Fb, Rb, blocks)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    out = {"point": T, "n_sessions": n, "n_boot": n_boot,
           "percentile": np.quantile(Tb, [lo_q, hi_q]).tolist()}
    # jackknife on the sample: acceleration and the sample SE
    Tj = _aggregate(F[None] - sf, R[None] - sr, blocks)
    d = Tj.mean() - Tj
    a = float((d ** 3).sum() / (6 * ((d ** 2).sum()) ** 1.5)) if np.any(d) else 0.0
    se = float(np.sqrt((n - 1) / n * ((Tj - Tj.mean()) ** 2).sum()))
    prop = (np.sum(Tb < T) + 0.5 * np.sum(Tb == T)) / n_boot
    z0 = float(norm.ppf(np.clip(prop, 1 / (n_boot + 1), n_boot / (n_boot + 1))))
    adj = [norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))) for z in norm.ppf([lo_q, hi_q])]
    out["bca"] = np.quantile(Tb, np.clip(adj, 0, 1)).tolist()
    out["bca_z0"], out["bca_acceleration"] = z0, a
    # studentized: jackknife SE inside each resample
    Tbj = _aggregate(Fb[:, None, :] - sf[None], Rb[:, None, :] - sr[None], blocks)  # B × n
    w = counts
    mean_b = (w * Tbj).sum(1) / n
    se_b = np.sqrt((n - 1) / n * (w * (Tbj - mean_b[:, None]) ** 2).sum(1))
    ok = se_b > 0
    t = (Tb[ok] - T) / se_b[ok]
    q_lo, q_hi = np.quantile(t, [lo_q, hi_q])
    out["bootstrap_t"] = [T - q_hi * se, T - q_lo * se]
    out["jackknife_se"] = se
    return out


def bootstrap_statistic(values: np.ndarray, groups: np.ndarray, stat, n_boot: int,
                        seed: int) -> np.ndarray:
    """Generic session bootstrap: ``stat`` receives the resampled row indices."""
    uniq, inv = np.unique(groups, return_inverse=True)
    rows = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        out.append(stat(np.concatenate([rows[k] for k in pick])))
    return np.asarray(out)


def self_test(seed: int = 0, n_sessions: int = 50, per_session: int = 300,
              n_boot: int = 1000, n_reps: int = 200) -> dict:
    """Checks the bootstrap on simulated clustered data with a known answer.

    1. **Unit.** Every resample consists of whole sessions.
    2. **Standard error.** For the mean of y_ij = u_i + e_ij (session effect u,
       SD 1; noise SD 2), the bootstrap SE of the grand mean is within 15% of the
       analytic cluster SE, sqrt(var(session means) / n_sessions).
    3. **Why the unit matters.** A pair-level bootstrap of the same data gives
       an SE far below the analytic one.
    4. **Coverage.** Across ``n_reps`` simulated data sets, the 95% percentile
       interval of the mean contains the true mean (0) at close to 95%.
    """
    rng = np.random.default_rng(seed)

    def simulate():
        g = np.repeat(np.arange(n_sessions), per_session)
        y = rng.normal(0, 1, n_sessions)[g] + rng.normal(0, 2, g.size)
        return y, g

    y, g = simulate()
    counts = resample_counts(n_sessions, n_boot, seed + 1)
    whole = bool(np.all(counts.sum(axis=1) == n_sessions)
                 and np.all(counts == np.round(counts)))
    sums, _ = session_sums(y[:, None], g)
    sizes, _ = session_sums(np.ones((y.size, 1)), g)
    boot_means = (counts @ sums[:, 0]) / (counts @ sizes[:, 0])
    se_boot = float(boot_means.std(ddof=1))
    means = sums[:, 0] / sizes[:, 0]
    se_analytic = float(means.std(ddof=1) / np.sqrt(n_sessions))
    pair_idx = rng.integers(0, y.size, size=(n_boot, y.size))
    se_pair = float(y[pair_idx].mean(axis=1).std(ddof=1))
    hits = 0
    for r in range(n_reps):
        yr, gr = simulate()
        s, _ = session_sums(yr[:, None], gr)
        c = resample_counts(n_sessions, 400, seed + 10 + r)
        bm = (c @ s[:, 0]) / (c.sum(axis=1) * per_session)
        lo, hi = np.quantile(bm, [0.025, 0.975])
        hits += lo <= 0.0 <= hi
    coverage = hits / n_reps
    mc_se = np.sqrt(0.95 * 0.05 / n_reps)
    checks = {
        "resamples_are_whole_sessions": whole,
        "se_within_15pct_of_analytic": abs(se_boot / se_analytic - 1) < 0.15,
        "pair_level_bootstrap_understates_se": se_pair < 0.5 * se_analytic,
        "coverage_95_within_3_mc_se": abs(coverage - 0.95) < 3 * mc_se,
    }
    return {"passed": all(checks.values()), "checks": checks, "se_bootstrap": se_boot,
            "se_analytic": se_analytic, "se_pair_level": se_pair, "coverage_95": coverage,
            "n_reps": n_reps}


__all__ = ["session_sums", "resample_counts", "paired_partial_r2", "interval_methods",
           "bootstrap_statistic", "self_test"]
