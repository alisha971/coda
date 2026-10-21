"""Projected detectability of the Layer-1 test at the planned scale (synthetic only).

G2 asks whether the minimum detectable effect (MDE) is small enough to matter.
Until real calibration (J3) runs, only a projection is possible. This module
injects a known partner → reply coupling into synthetic data with the planned
shape and measures, with the actual ridge estimator and a donor null, how often
the test detects it.

**Assumptions (each one can make the projection optimistic or pessimistic):**

- Z is Gaussian with the planned width (text components, metadata and one
  indicator per training replier); real text components are not Gaussian;
- reply residuals have a session-level component (``session_sd``) and, by
  default, are otherwise independent over turns; ``resid_rho`` adds AR(1)
  serial dependence within each session;
- partner delivery A is independent of Z apart from a fixed loading
  (``a_on_z``); ``a_rho`` adds AR(1) serial dependence within each session;
- by default there is no state shared by A and Y; ``drift_sd`` adds one
  AR(1) state per session (persistence ``drift_rho``) to both, which is
  genuine predictive dependence at β = 0;
- donors are partner vectors from other sessions in the same split, drawn at
  random (the N1 form without duration and word-count matching);
- the injected coupling is the identity map on matched dimensions, scaled per
  block to the target partial R² as in J3.

The projection therefore states how detectable an effect of each size would be
**if** the real data resembled these assumptions. It is not the registered MDE,
which only J3 on the real sample provides.
"""

from __future__ import annotations

import numpy as np

from src.controls.injection import (
    betas_for_target,
    mde_from_power,
    rejection_rate,
    residualise_on_z,
    session_block_residuals,
)
from src.evaluation.metrics import block_partial_r2_from_sse
from src.models.ridge import RidgeEngine, ZCache, block_col_groups

BLOCK_SIZES = [2, 1, 2, 3]


def _blocks():
    out, k = [], 0
    for s in BLOCK_SIZES:
        out.append(np.arange(k, k + s))
        k += s
    return out


def _ar1(rng, n_sessions: int, length: int, width: int, rho: float) -> np.ndarray:
    """Unit-variance AR(1) series per session, stacked session-major."""
    e = rng.normal(size=(n_sessions, length, width))
    if rho == 0:
        return e.reshape(-1, width)
    x = np.empty_like(e)
    x[:, 0] = e[:, 0]
    c = np.sqrt(1 - rho ** 2)
    for t in range(1, length):
        x[:, t] = rho * x[:, t - 1] + c * e[:, t]
    return x.reshape(-1, width)


def synthetic_design(n_sessions: int, pairs_per_session: int, z_dense: int, seed: int,
                     session_sd: float = 0.3, a_on_z: float = 0.2,
                     train_frac: float = 0.7, test_frac: float = 0.2,
                     resid_rho: float = 0.0, a_rho: float = 0.0,
                     drift_sd: float = 0.0, drift_rho: float = 0.95) -> dict:
    """Rows are session-major and in turn order within each session."""
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_sessions), pairs_per_session)
    n = g.size
    order = rng.permutation(n_sessions)
    n_tr, n_te = int(train_frac * n_sessions), int(test_frac * n_sessions)
    split = np.empty(n_sessions, dtype=object)
    split[order[:n_tr]] = "train"
    split[order[n_tr:n_tr + n_te]] = "test"
    split[order[n_tr + n_te:]] = "val"
    s = split[g]
    Zd = rng.normal(size=(n, z_dense))
    # one indicator per training replier; two repliers per session
    replier = g * 2 + rng.integers(0, 2, n)
    train_rep = np.unique(replier[s == "train"])
    col = {r: j for j, r in enumerate(train_rep)}
    Zr = np.zeros((n, len(train_rep)), np.float32)
    for i, r in enumerate(replier):
        j = col.get(r)
        if j is not None:
            Zr[i, j] = 1.0
    Z = np.hstack([Zd, Zr])
    shared = (drift_sd * _ar1(rng, n_sessions, pairs_per_session, 1, drift_rho)
              if drift_sd else np.zeros((n, 1)))
    A = _ar1(rng, n_sessions, pairs_per_session, 8, a_rho) + a_on_z * Zd[:, :8] + shared
    fz = Zd[:, :16] @ rng.normal(size=(16, 8)) * 0.15 + shared

    def noise():   # fresh residuals with the same session clustering and serial dependence
        return (_ar1(rng, n_sessions, pairs_per_session, 8, resid_rho)
                + session_sd * rng.normal(size=(n_sessions, 8))[g])

    eps = noise()
    return {"Z": Z, "A": A, "fhat": fz, "eps": eps, "noise": noise, "groups": g, "split": s,
            "rng": rng}


def project(n_sessions: int = 250, pairs_per_session: int = 300, z_dense: int = 132,
            targets=(0.0, 0.0025, 0.005, 0.01), replicates: int = 40, draws: int = 39,
            alpha: float = 0.05, power: float = 0.80, seed: int = 0, folds: int = 3,
            lambda_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0), noise_mode: str = "fresh",
            **design) -> dict:
    """Power at each injected partial R² and the projected MDE.

    ``noise_mode`` sets how replicate residuals are made:

    - ``fresh``: new draws from the generating process (the true power);
    - ``j3_rows`` / ``j3_ordered``: J3's procedure. One residual realisation on
      the training rows is the pool, and each replicate resamples it with
      :func:`session_block_residuals` (unordered or ordered). Comparing these
      with ``fresh`` shows whether J3's resampling over- or under-states power.

    ``design`` passes through to :func:`synthetic_design` (``resid_rho``,
    ``a_rho``, ``drift_sd``, ``drift_rho``, ``session_sd``).
    """
    if noise_mode not in ("fresh", "j3_rows", "j3_ordered"):
        raise ValueError(f"unknown noise mode {noise_mode!r}")
    d = synthetic_design(n_sessions, pairs_per_session, z_dense, seed, **design)
    tr, te = d["split"] == "train", d["split"] == "test"
    zc = ZCache(d["Z"], d["groups"], tr, te, folds, seed)
    blocks = _blocks()
    g_tilde = residualise_on_z(d["A"], d["Z"][:, :z_dense], tr)
    rng = d["rng"]
    sess_tr = np.unique(d["groups"][tr])
    sess_te = np.unique(d["groups"][te])
    out = {"assumptions": {"n_sessions": n_sessions, "pairs_per_session": pairs_per_session,
                           "z_columns": int(d["Z"].shape[1]), "replicates": replicates,
                           "draws": draws, "alpha": alpha, "power_level": power,
                           "noise_mode": noise_mode, **design},
           "levels": []}
    if noise_mode == "fresh":
        make_eps = d["noise"]
    else:
        pool, pool_g = d["eps"][tr], d["groups"][tr]

        def make_eps():
            return session_block_residuals(pool, pool_g, d["groups"], rng,
                                           ordered=noise_mode == "j3_ordered")
    for t in targets:
        beta = betas_for_target(t, g_tilde, d["eps"][tr], blocks, tr)
        Y = np.hstack([d["fhat"] + d["A"] * beta + make_eps() for _ in range(replicates)])
        eng = RidgeEngine(zc, Y, block_col_groups(BLOCK_SIZES, replicates), lambda_grid)
        red = eng.run(None, eval_sse_only=True).extra["eval_sse"]

        def stat(A, eng=eng, red=red):
            full = eng.run(A, eval_sse_only=True).extra["eval_sse"]
            return np.array([block_partial_r2_from_sse(full[r * 8:(r + 1) * 8],
                                                       red[r * 8:(r + 1) * 8], blocks).mean()
                             for r in range(replicates)])

        obs = stat(d["A"])
        null = []
        for _ in range(draws):
            Ad = d["A"].copy()
            for sess in (sess_tr, sess_te):           # donors from other sessions, same split
                rows = np.flatnonzero(np.isin(d["groups"], sess))
                Ad[rows] = d["A"][rows[np.roll(np.argsort(d["groups"][rows] +
                                                          rng.random(len(rows))),
                                               pairs_per_session)]]
            null.append(stat(Ad))
        null = np.stack(null)                         # draws × replicates
        p = (null >= obs[None]).mean(axis=0)
        out["levels"].append({"target_partial_r2": t,
                              "realised_mean": float(obs.mean()),
                              "rejection": rejection_rate(p, alpha)})
    t_pos = [x["target_partial_r2"] for x in out["levels"] if x["target_partial_r2"] > 0]
    pw = [x["rejection"]["rate"] for x in out["levels"] if x["target_partial_r2"] > 0]
    out["mde_projected"] = mde_from_power(t_pos, pw, power)
    return out


__all__ = ["synthetic_design", "project"]
