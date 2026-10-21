"""Can the design separate local partner → reply coupling from shared drift? (synthetic study)

H4 contrasts the real partner turn with N2 donors: the same partner speaker, same
interaction, at least ten turns away. N2 holds speaker, dyad, channel, topic and
register fixed. It does **not** hold time fixed: the real partner turn is one
turn before the reply, every N2 donor at least ten. Any shared state that changes
over a few turns (mood, topic phase, mutual arousal, recording conditions)
therefore makes the adjacent turn more predictive than a distant one even when
no speaker responds to the other's delivery. That is the shared-drift
alternative this module tests, on synthetic corpora only.

Three analyses are computed on one corpus, all with the Layer-1 estimator:

1. **Registered.** Δ of ``plus_partner_delivery`` over ``plus_partner_text``
   against N1 and N2 donor nulls; the G3 ordering rule of the decision file.
2. **Distance profile (candidate, descriptive).** The same null rebuilt with
   donors at exact turn distances 2, 4 and 6 (same partner speaker, before or
   after). Under shared drift the null statistic should fall gradually with
   distance; under purely local coupling every donor distance should look like
   no information.
3. **History-conditioned test (candidate).** The registered N2 comparison inside
   a model whose fixed part also holds recent history: the replier's previous
   turn (S) and the partner's previous turn (two turns before the partner turn).
   If drift is slow, recent history carries most of the drift state.

None of these is registered. This module exists to measure how they behave.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from src.config import RidgeConfig
from src.controls.donors import draw_n1, draw_n2
from src.evaluation.metrics import block_partial_r2_from_sse
from src.experiments.pipeline import AnalysisData
from src.models.ridge import RidgeEngine, ZCache
from src.targets.normalise import exclusion_matrix, mode_for_split, standardise_donors


def _main_index(turns):
    main = turns[~turns["embedded"].astype(bool)]
    return {(r.interaction_key, int(r.turn_index)): r.turn_id
            for r in main.itertuples(index=False)}


def partner_history(data: AnalysisData, turns, usable: set) -> tuple[np.ndarray, np.ndarray]:
    """Standardised delivery of the partner's previous turn (two turns before the partner turn)."""
    idx = _main_index(turns)
    t = turns.set_index("turn_id")
    ids, ok = [], []
    for r in data.pairs.itertuples(index=False):
        k = int(t.at[r.partner_turn_id, "turn_index"])
        tid = idx.get((r.interaction_key, k - 2))
        good = tid is not None and tid in usable
        ids.append(tid if good else r.partner_turn_id)
        ok.append(good)
    ex = exclusion_matrix(data.pairs)
    H = data.standardiser.rows(ids, mode_for_split(data.pairs["split"]), ex)
    vals = H[data.standardiser.dims].to_numpy()
    ok = np.asarray(ok) & np.isfinite(vals).all(axis=1)
    return vals, ok


def exact_distance_donors(data: AnalysisData, turns, usable: set, distance: int, n_draws: int,
                          seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Donors from the partner speaker exactly ``distance`` turns away (before or after)."""
    if distance % 2:
        raise ValueError("same-speaker turns lie at even distances")
    idx = _main_index(turns)
    t = turns.set_index("turn_id")
    rng = np.random.default_rng(seed)
    out = np.empty((len(data.pairs), n_draws), dtype=object)
    ok = np.zeros(len(data.pairs), bool)
    for i, r in enumerate(data.pairs.itertuples(index=False)):
        k = int(t.at[r.partner_turn_id, "turn_index"])
        cand = [idx.get((r.interaction_key, k + s * distance)) for s in (-1, 1)]
        cand = [c for c in cand if c is not None and c in usable and c != r.reply_turn_id]
        if cand:
            out[i] = np.asarray(cand, dtype=object)[rng.integers(0, len(cand), n_draws)]
            ok[i] = True
        else:
            out[i] = r.partner_turn_id      # placeholder; the row is excluded via ``ok``
    return out, ok


def increment_and_nulls(data: AnalysisData, Z: np.ndarray, rows: np.ndarray,
                        donor_sets: dict[str, np.ndarray], cfg: RidgeConfig) -> dict:
    """Observed aggregate Δ of adding A to ``Z`` and its donor-null distributions."""
    zc = ZCache(Z, data.groups, data.train & rows, data.test & rows, cfg.nested_cv_folds)
    eng = RidgeEngine(zc, data.Y, data.blocks, cfg.lambda_grid)
    red = eng.run(None, eval_sse_only=True).extra["eval_sse"]

    def stat(A):
        full = eng.run(A, eval_sse_only=True).extra["eval_sse"]
        return float(block_partial_r2_from_sse(full, red, data.blocks).mean())

    obs = stat(data.A)
    out = {"observed": obs, "n_pairs": int(rows.sum())}
    for name, donors in donor_sets.items():
        null = np.array([stat(standardise_donors(data.pairs, donors[:, [k]],
                                                 data.standardiser)[:, 0, :])
                         for k in range(donors.shape[1])])
        out[name] = {"p": float((null >= obs).mean()), "null_mean": float(null.mean()),
                     "null_median": float(np.median(null)),
                     "null_q95": float(np.quantile(null, 0.95))}
    return out


def registered_reading(res: dict, alpha: float = 0.05) -> str:
    """The G3 ordering as currently coded (the registered ordering rule)."""
    gt1 = res["N1"]["p"] <= alpha
    gt2 = res["N2"]["p"] <= alpha
    n2_gt_n1 = res["N2"]["null_median"] > res["N1"]["null_q95"]
    if gt1 and gt2 and n2_gt_n1:
        return "real>N2>N1"
    if gt1 and not gt2 and n2_gt_n1:
        return "real~N2>N1"
    if not gt1:
        return "real~N1"
    return "real>N1,real>N2,N2~N1" if gt2 else "real>N1,real~N2,N2~N1"


def scenario_run(params: dict, seed: int, n_draws: int = 39, n_draws_profile: int = 19,
                 n_sessions: int = 40, turns: int = 80, cfg: RidgeConfig | None = None,
                 distances=(2, 4, 6)) -> dict:
    """All three analyses on one synthetic corpus."""
    from src.experiments import dry_run as dr

    cfg = cfg or RidgeConfig(nested_cv_folds=3)
    c = dr.make_corpus(n_sessions, turns, seed=seed, **params)
    data, prosody, _ = dr.prepare(c, Path(tempfile.mkdtemp(prefix="coda_h4_")))
    usable = set(prosody.loc[prosody["usable"].astype(bool), "turn_id"])
    H, h_ok = partner_history(data, c.turns, usable)
    prof = {d: exact_distance_donors(data, c.turns, usable, d, n_draws_profile, seed + d)
            for d in distances}
    rows = h_ok.copy()
    for _, ok in prof.values():
        rows &= ok
    n1, _ = draw_n1(data.pairs, c.turns, n_draws, seed)
    n2 = draw_n2(data.pairs, c.turns, usable, n_draws, seed + 1)
    Z = data.Z.full()
    reg = increment_and_nulls(data, Z, rows, {"N1": n1, "N2": n2}, cfg)
    profile = increment_and_nulls(data, Z, rows, {f"d{d}": prof[d][0] for d in distances}, cfg)
    cond = increment_and_nulls(data, np.hstack([Z, data.S, H]), rows,
                               {"N2": n2, "d2": prof[2][0]}, cfg)
    return {"params": params, "seed": seed, "n_pairs": int(rows.sum()),
            "registered": reg, "reading": registered_reading(reg),
            "profile": {k: v for k, v in profile.items()},
            "conditioned": cond}


SCENARIOS = {
    "S1_local_only": {"coupling": 0.4},
    "S2_drift_only_rho0.95": {"coupling": 0.0, "drift_sd": 1.0, "drift_rho": 0.95},
    "S2_drift_only_rho0.80": {"coupling": 0.0, "drift_sd": 1.0, "drift_rho": 0.80},
    "S2_drift_only_rho0.99": {"coupling": 0.0, "drift_sd": 1.0, "drift_rho": 0.99},
    "S3_local_and_drift": {"coupling": 0.4, "drift_sd": 1.0, "drift_rho": 0.95},
    "S4_neither": {"coupling": 0.0},
    "S5a_interaction_offsets": {"coupling": 0.0, "n_interactions": 4, "interaction_sd": 1.0},
    "S5b_self_momentum": {"coupling": 0.0, "self_ar": 0.5},
    "S5c_self_momentum_and_drift": {"coupling": 0.0, "self_ar": 0.5, "drift_sd": 1.0,
                                    "drift_rho": 0.95},
}


__all__ = ["partner_history", "exact_distance_donors", "increment_and_nulls",
           "registered_reading", "scenario_run", "SCENARIOS"]
