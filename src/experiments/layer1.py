"""Layer-1 computations shared by the J1-J4 drivers.

Each function takes an :class:`~src.experiments.pipeline.AnalysisData` and
returns plain arrays and dictionaries; the drivers (``run_j1.py`` ...) decide
what is written where. The conditions, read from ``configs/j1_conditions.yaml``:

| Condition | Design | Answers |
|---|---|---|
| ``Z_content`` | Z without partner text | how predictable delivery is at all |
| ``plus_partner_text`` | Z | what the partner transcript adds |
| ``plus_partner_delivery`` | Z + A | what partner delivery adds (H1) |
| ``plus_self_previous`` | Z + S | the self-momentum competitor |

The H1 increment is ``plus_partner_delivery`` over ``plus_partner_text``; the
donor nulls replace A in that comparison and keep everything else, including
the reduced model's fit (Z does not change under a donor).
"""

from __future__ import annotations

import numpy as np

from src.config import RidgeConfig
from src.controls.injection import coupling_matrix, injected_targets, rejection_rate
from src.evaluation.bootstrap import interval_methods, paired_partial_r2
from src.evaluation.metrics import (
    block_partial_r2,
    block_partial_r2_from_sse,
    interval_coverage,
    loglik_increment,
    per_pair_errors,
    sse_by_column,
)
from src.experiments.pipeline import AnalysisData
from src.models.ridge import RidgeEngine, ZCache, block_col_groups
from src.models.variance import fit_variance
from src.statistics.permutation import donor_p_value
from src.targets.normalise import standardise_donors
from src.targets.orthogonalise import BlockOrthogonaliser

COMPARISONS = {
    "partner_text_over_content": ("plus_partner_text", "Z_content"),
    "partner_delivery_over_partner_text": ("plus_partner_delivery", "plus_partner_text"),
    "self_previous_over_partner_text": ("plus_self_previous", "plus_partner_text"),
}


def caches(data: AnalysisData, cfg: RidgeConfig, rows: np.ndarray | None = None,
           eval_split: str = "test") -> dict[str, ZCache]:
    """Z-only precomputation for the content and full conditioning sets."""
    rows = np.ones(len(data.pairs), bool) if rows is None else rows
    fit = data.train & rows
    ev = (data.split == eval_split) & rows
    return {"content": ZCache(data.Z.content(), data.groups, fit, ev, cfg.nested_cv_folds),
            "full": ZCache(data.Z.full(), data.groups, fit, ev, cfg.nested_cv_folds)}


def fit_j1(data: AnalysisData, cfg: RidgeConfig, settings, rows: np.ndarray | None = None,
           n_boot: int | None = None, seed: int = 20260912) -> dict:
    """The four conditions, their increments, intervals and both units."""
    zc = caches(data, cfg, rows)
    blocks = data.blocks
    grid = cfg.lambda_grid
    eng_c = RidgeEngine(zc["content"], data.Y, blocks, grid)
    eng_f = RidgeEngine(zc["full"], data.Y, blocks, grid)
    fits = {"Z_content": (eng_c.run(None, return_oof=True, return_edf=True), None, zc["content"]),
            "plus_partner_text": (eng_f.run(None, return_oof=True, return_edf=True), None,
                                  zc["full"]),
            "plus_partner_delivery": (eng_f.run(data.A, return_oof=True, return_edf=True),
                                      data.A, zc["full"]),
            "plus_self_previous": (eng_f.run(data.S, return_oof=True, return_edf=True), data.S,
                                   zc["full"])}
    ev = zc["full"].eval_idx
    fit_idx = zc["full"].fit_idx
    y = data.Y[ev]
    var = {}
    for name, (res, A, z) in fits.items():
        resid = data.Y[fit_idx] - res.oof_fit
        var[name] = fit_variance(z, resid, A, grid,
                                 settings.variance_model["floor_fraction_of_mean_squared_residual"])
    md11 = settings.bootstrap
    n_boot = n_boot or int(md11["resamples"])
    out = {"conditions": {}, "comparisons": {}, "n_eval_pairs": int(len(ev)),
           "n_eval_sessions": int(len(np.unique(data.groups[ev]))),
           "n_fit_pairs": int(len(fit_idx)), "errors": {}, "predictions": {}}
    for name, (res, _, _) in fits.items():
        cov = interval_coverage(y, res.pred_eval, var[name].var_eval, blocks, md11["levels"])
        out["conditions"][name] = {
            "lambda": res.lambdas.tolist(), "lambda_rel": res.lambda_rel.tolist(),
            "lambda_at_grid_edge": res.at_grid_edge.tolist(), "edf": res.edf.tolist(),
            "variance_floor_share": var[name].floored_share,
            "coverage": {str(k): v.tolist() for k, v in cov.items()}}
        out["errors"][name] = per_pair_errors(y, res.pred_eval)
        out["predictions"][name] = res.pred_eval
    for comp, (full, red) in COMPARISONS.items():
        pr2 = block_partial_r2(y, fits[full][0].pred_eval, fits[red][0].pred_eval, blocks)
        ll = loglik_increment(y, fits[full][0].pred_eval, var[full].var_eval,
                              fits[red][0].pred_eval, var[red].var_eval, blocks)
        boot = paired_partial_r2(out["errors"][full], out["errors"][red], data.groups[ev],
                                 blocks, n_boot, seed, md11["levels"])
        # The aggregate 95% interval by the selected method, with the other two
        # methods kept beside it (docs/research/simulations/BOOTSTRAP_COMPARISON_PROTOCOL.md).
        # Same resamples.
        im = interval_methods(out["errors"][full], out["errors"][red], data.groups[ev],
                              blocks, n_boot, seed, 0.95)
        method = md11.get("interval", "percentile")
        out["comparisons"][comp] = {
            "block_partial_r2": pr2.tolist(), "aggregate_partial_r2": float(pr2.mean()),
            "block_loglik_increment": ll.tolist(), "aggregate_loglik_increment": float(ll.mean()),
            "units_agree_in_sign": bool(np.sign(pr2.mean()) == np.sign(ll.mean())),
            "intervals": boot["intervals"], "n_boot": boot["n_boot"],
            "block_interval_method": "percentile",
            "aggregate_interval_95": {"method": method, "interval": im[method],
                                      "simulated_worst_case_coverage":
                                          md11.get("simulated_worst_case_coverage"),
                                      "all_methods": {m: im[m] for m in
                                                      ("percentile", "bca", "bootstrap_t")}}}
    out["_engine_full"] = eng_f
    out["_reduced_pred"] = fits["plus_partner_text"][0].pred_eval
    return out


def donor_null(data: AnalysisData, engine_full: RidgeEngine, reduced_pred: np.ndarray,
               donors: np.ndarray, rows: np.ndarray | None = None) -> dict:
    """Refit ``plus_partner_delivery`` with each donor draw in place of A.

    ``donors`` is (n_pairs × B) turn ids. Each draw is standardised exactly as
    the partner it replaces, then the full model is refitted with its own λ
    search; the reduced model is the real ``plus_partner_text`` fit.
    """
    ev = engine_full.zc.eval_idx
    sse_red = sse_by_column(data.Y[ev], reduced_pred)
    B = donors.shape[1]
    blocks = np.empty((B, len(data.blocks)))
    sub = data.pairs if rows is None else data.pairs[rows]
    for k in range(B):
        Ad = np.full_like(data.A, np.nan)
        vec = standardise_donors(sub, donors[:, [k]], data.standardiser)[:, 0, :]
        if rows is None:
            Ad[:] = vec
        else:
            Ad[rows] = vec
        bad = ~np.isfinite(Ad[engine_full.zc.fit_idx]).all(axis=1)
        if bad.any():
            raise ValueError(f"donor draw {k}: {int(bad.sum())} training donors could not be "
                             "standardised; donors must come from usable turns")
        res = engine_full.run(Ad, eval_sse_only=True)
        blocks[k] = block_partial_r2_from_sse(res.extra["eval_sse"], sse_red, data.blocks)
    return {"block": blocks, "aggregate": blocks.mean(axis=1)}


def h1_h4_tests(observed: dict, null_n1: dict, null_n2: dict) -> dict:
    """Primary test (vs N1) and H4 (vs N2), aggregate and per block."""
    obs_agg = observed["aggregate_partial_r2"]
    obs_blk = np.asarray(observed["block_partial_r2"])
    return {"H1_vs_N1": donor_p_value(obs_agg, null_n1["aggregate"]),
            "H4_vs_N2": donor_p_value(obs_agg, null_n2["aggregate"]),
            "block_vs_N1": [donor_p_value(obs_blk[b], null_n1["block"][:, b])
                            for b in range(len(obs_blk))],
            "block_vs_N2": [donor_p_value(obs_blk[b], null_n2["block"][:, b])
                            for b in range(len(obs_blk))]}


# --------------------------------------------------------------------------
# J3: calibration
# --------------------------------------------------------------------------


def training_baseline(data: AnalysisData, cfg: RidgeConfig) -> dict:
    """Content-only predictability from out-of-fold predictions inside training.

    Used for the baseline report before J1, so that the test split is touched
    only once per condition: R² per block from grouped inner-CV
    predictions on training pairs, never from test pairs.
    """
    zc = caches(data, cfg)["content"]
    res = RidgeEngine(zc, data.Y, data.blocks, cfg.lambda_grid).run(None, return_oof=True,
                                                                     return_edf=True)
    y = data.Y[zc.fit_idx]
    ss_res = ((y - res.oof_fit) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    r2 = np.array([1 - ss_res[b].sum() / ss_tot[b].sum() for b in data.blocks])
    return {"block_oof_r2": r2.tolist(), "aggregate_oof_r2": float(r2.mean()),
            "lambda_rel": res.lambda_rel.tolist(), "lambda_at_grid_edge": res.at_grid_edge.tolist(),
            "edf": res.edf.tolist(), "n_training_pairs": int(len(zc.fit_idx)),
            "note": "training out-of-fold only; the test split is not read"}


def content_fit_for_injection(data: AnalysisData, cfg: RidgeConfig,
                              source: str = "content") -> dict:
    """f̂ for every pair, and the out-of-fold residual pool.

    ``source="content"`` (registered) predicts from Z_content. ``source="full"``
    predicts from Z_content plus partner text, so β = 0 replicates also carry
    any reply dependence on partner words that the text components capture
    (a sensitivity run, not a replacement for the registered G1).
    """
    if source not in ("content", "full"):
        raise ValueError(f"unknown content source {source!r}")
    zc = caches(data, cfg)[source]
    res = RidgeEngine(zc, data.Y, data.blocks, cfg.lambda_grid).run(None, return_oof=True)
    fhat = np.full_like(data.Y, np.nan)
    fhat[zc.fit_idx] = res.oof_fit
    fhat[zc.eval_idx] = res.pred_eval
    pool = data.Y[zc.fit_idx] - res.oof_fit
    return {"fhat": fhat, "pool": pool, "pool_groups": data.groups[zc.fit_idx],
            "rows": np.isfinite(fhat).all(axis=1)}


def j3_level(data: AnalysisData, cfg: RidgeConfig, inj: dict, target_pr2: float,
             coupling: str, cross_map: dict, n_replicates: int,
             donors: dict[str, np.ndarray], alpha: float, seed: int,
             resampling: str = "session_block", rows: np.ndarray | None = None,
             replicate_ids=None) -> dict:
    """One β level: R replicates of Y*, each tested against N1 and N2 donor nulls.

    ``resampling`` is the residual scheme (``session_block`` or
    ``session_block_ordered``). ``rows`` restricts the level to a subset of
    pairs (a function stratum for per-stratum calibration).
    """
    if resampling not in ("session_block", "session_block_ordered"):
        raise ValueError(f"unknown residual resampling {resampling!r}")
    rows = inj["rows"] if rows is None else (inj["rows"] & rows)
    if replicate_ids is not None:
        replicate_ids = [int(r) for r in replicate_ids]
        n_replicates = len(replicate_ids)
    G = coupling_matrix(coupling, cross_map)
    train_rows = data.train & rows
    Ystar, beta = injected_targets(inj["fhat"], data.A, data.Z.full(), inj["pool"],
                                   inj["pool_groups"], data.groups, train_rows, target_pr2, G,
                                   data.blocks, n_replicates, seed,
                                   ordered=resampling == "session_block_ordered",
                                   replicate_ids=replicate_ids)
    m = data.Y.shape[1]
    for r in range(n_replicates):                      # orthogonalisation refitted on Y*
        cols = slice(r * m, (r + 1) * m)
        o = BlockOrthogonaliser().fit(Ystar[train_rows, cols], np.flatnonzero(train_rows), "j3")
        Ystar[:, cols] = o.transform(Ystar[:, cols])
    zc = ZCache(data.Z.full(), data.groups, train_rows, data.test & rows, cfg.nested_cv_folds)
    groups = block_col_groups([len(b) for b in data.blocks], n_replicates)
    eng = RidgeEngine(zc, Ystar, groups, cfg.lambda_grid)
    red = eng.run(None, eval_sse_only=True).extra["eval_sse"]
    full = eng.run(data.A, eval_sse_only=True).extra["eval_sse"]

    def per_rep(sse):
        return np.array([block_partial_r2_from_sse(sse[r * m:(r + 1) * m],
                                                   red[r * m:(r + 1) * m], data.blocks)
                         for r in range(n_replicates)])            # (R × n_blocks)

    obs = per_rep(full)
    out = {"target_partial_r2": target_pr2, "coupling": coupling, "beta": beta.tolist(),
           "replicates": n_replicates, "observed_block": obs.tolist(),
           "replicate_ids": (list(range(n_replicates)) if replicate_ids is None
                             else replicate_ids),
           "realised_aggregate_mean": float(obs.mean())}
    for scheme, d in donors.items():
        null = []
        for k in range(d.shape[1]):
            Ad = standardise_donors(data.pairs, d[:, [k]], data.standardiser)[:, 0, :]
            null.append(per_rep(eng.run(Ad, eval_sse_only=True).extra["eval_sse"]))
        null = np.stack(null)                                       # (B × R × n_blocks)
        p_agg = (null.mean(axis=2) >= obs.mean(axis=1)[None, :]).mean(axis=0)
        p_blk = (null >= obs[None]).mean(axis=0)
        out[scheme] = {"p_aggregate": p_agg.tolist(), "p_block": p_blk.tolist(),
                       "rejection_aggregate": rejection_rate(p_agg, alpha),
                       "rejection_block": [rejection_rate(p_blk[:, b], alpha)
                                           for b in range(p_blk.shape[1])]}
    return out


def merge_level_chunks(chunks: list[dict], alpha: float) -> dict:
    """One β level from its replicate chunks; identical to computing it unchunked."""
    from src.controls.injection import rejection_rate

    first = chunks[0]
    obs = np.concatenate([np.asarray(c["observed_block"]) for c in chunks])
    out = {"target_partial_r2": first["target_partial_r2"], "coupling": first["coupling"],
           "beta": first["beta"], "replicates": int(len(obs)), "observed_block": obs.tolist(),
           "replicate_ids": [r for c in chunks for r in c["replicate_ids"]],
           "realised_aggregate_mean": float(obs.mean()), "chunks": len(chunks)}
    for scheme in ("N1", "N2"):
        if scheme not in first:
            continue
        p_agg = np.concatenate([np.asarray(c[scheme]["p_aggregate"]) for c in chunks])
        p_blk = np.concatenate([np.asarray(c[scheme]["p_block"]) for c in chunks])
        out[scheme] = {"p_aggregate": p_agg.tolist(), "p_block": p_blk.tolist(),
                       "rejection_aggregate": rejection_rate(p_agg, alpha),
                       "rejection_block": [rejection_rate(p_blk[:, b], alpha)
                                           for b in range(p_blk.shape[1])]}
    return out


def stratum_mde(data: AnalysisData, cfg: RidgeConfig, inj: dict,
                strata: dict[str, np.ndarray], targets: list[float], n_replicates: int,
                donors: dict[str, np.ndarray], alpha: float, power_level: float, seed: int,
                resampling: str = "session_block",
                coupling: str = "identity_matched_dimensions", cross_map: dict | None = None,
                ) -> dict:
    """Per-stratum detectability for G5: the J3 injection restricted to
    each stratum's pairs, with the stratum's own MDE from its N1 power curve.

    No scaling from the whole-sample MDE is assumed; every stratum is
    calibrated directly, at the cost of one J3 sweep per stratum.
    """
    from src.controls.injection import mde_from_power

    out = {}
    for k, (name, rows) in enumerate(strata.items()):
        lv = [j3_level(data, cfg, inj, t, coupling, cross_map or {}, n_replicates, donors,
                       alpha, seed + 100 * k + i, resampling=resampling, rows=rows)
              for i, t in enumerate(targets)]
        pw = [x["N1"]["rejection_aggregate"]["rate"] for x in lv]
        realised = [x["realised_aggregate_mean"] for x in lv]
        m_t = mde_from_power(targets, pw, power_level)
        m_r = mde_from_power(realised, pw, power_level)
        both = [v for v in (m_t, m_r) if v is not None]
        out[name] = {"pairs": int((inj["rows"] & rows).sum()), "targets": list(targets),
                     "power": pw, "realised": realised,
                     "mde": max(both) if len(both) == 2 else None}
    return out


__all__ = ["COMPARISONS", "caches", "fit_j1", "donor_null", "h1_h4_tests",
           "content_fit_for_injection", "j3_level", "merge_level_chunks",
           "stratum_mde"]
