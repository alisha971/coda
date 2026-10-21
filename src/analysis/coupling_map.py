"""The H2 coupling map and its reproducibility battery (H2).

Read from J1's ``plus_partner_delivery`` condition with no new scientific
fitting: reduced-rank ridge factors of the partner → reply coefficient block.
The six registered requirements and how each is computed:

1. **Reduced-rank structure** — the rank selected by grouped inner CV and the
   share of fitted coupling each direction carries.
2. **Session-bootstrap stability** — training sessions resampled with
   replacement; RRR refitted at the reference λ and rank; each reference
   direction matched to the resampled direction with the largest absolute
   cosine (reply loadings), signs aligned; the share of resamples in which each
   partner loading keeps its reference sign, and the loading SD.
3. **Held-out-speaker-fold stability** — RRR refitted leaving out each
   training fold of sessions (speakers do not recur across sessions, so each
   fold holds out its speakers); absolute cosine with the reference direction.
4. **Concentration** — share of the leading r directions in Σd².
5. **Null comparison** — the leading strength d₁ against the same statistic
   refitted under the N1 donor draws. A random orthogonal rotation of A would
   be uninformative here: ridge with an isotropic penalty is rotation-invariant
   in A, so the spectrum of d would not change. The donor refit is the
   "permutation null" alternative that the design also names.
6. **Layer consistency** — rank agreement between the partner dimensions the
   coupling map loads on and the per-dimension probe R² of the neural partner
   representation; computed when the neural arm has run.

**Verdict.** H2's null is that no direction survives (2), (3) and (5) together.
"Survives" needs numerical thresholds for (2) and (3), which no governing
document states. Until they are set, the battery is computed
and reported and the verdict is ``requires_approved_thresholds``.
"""

from __future__ import annotations

import numpy as np

from src.models.ridge import RidgeEngine, ZCache
from src.models.rrr import RRRResult, fit_rrr, leading_strength
from src.statistics.permutation import donor_p_value


def _match(ref_Q: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each reference column, the best |cos| column of Q, its index and sign."""
    c = ref_Q.T @ Q / (np.linalg.norm(ref_Q, axis=0)[:, None] * np.linalg.norm(Q, axis=0)[None])
    j = np.abs(c).argmax(axis=1)
    cos = c[np.arange(len(j)), j]
    return np.abs(cos), j, np.sign(cos)


def _refit(Z, A, Y, groups, rows, blocks, lam_rel, rank) -> RRRResult:
    """RRR on the given rows (duplicates allowed) at fixed λ and rank."""
    Zr, Ar, Yr = Z[rows], A[rows], Y[rows]
    g = np.arange(len(rows))           # resampled rows are distinct fitting rows
    fit = np.ones(len(rows), bool)
    zc = ZCache(Zr, g, fit, np.zeros(len(rows), bool), n_folds=0)
    eng = RidgeEngine(zc, Yr, blocks, lambda_rel_grid=(lam_rel,))
    return fit_rrr(eng, Ar, blocks, fixed=(lam_rel, rank))


def subspace_similarity(Q1: np.ndarray, Q2: np.ndarray) -> float:
    """Mean squared cosine of the principal angles between two column spaces.

    1 means the same subspace, 0 orthogonal. For one column it is the squared
    Tucker congruence (cos²). For several columns it does not penalise
    directions that rotate inside a stable plane, which single-direction
    matching does when two directions are about equally strong.
    """
    a, _ = np.linalg.qr(np.asarray(Q1, dtype=np.float64))
    b, _ = np.linalg.qr(np.asarray(Q2, dtype=np.float64))
    r = min(a.shape[1], b.shape[1])
    return float(np.linalg.norm(a.T @ b) ** 2 / r)


def stability_statistics(Z, A, Y, groups, train_rows, blocks, lam_rel: float, rank: int,
                         ref_Q: np.ndarray, ref_P: np.ndarray, fold_of_row: np.ndarray,
                         n_boot: int, seed: int) -> dict:
    """Requirements (2) and (3) as subspace similarities of reply and partner loadings.

    (2) is the mean over session-bootstrap refits; (3) the minimum over
    leave-one-fold-out refits (folds hold out whole sessions, hence their
    speakers). λ and rank stay those of the reference fit.
    """
    tr = np.flatnonzero(train_rows)
    sessions = np.unique(groups[tr])
    rows_of = {s: tr[groups[tr] == s] for s in sessions}
    rng = np.random.default_rng(seed)
    bq, bp = [], []
    for _ in range(n_boot):
        pick = rng.choice(sessions, size=len(sessions), replace=True)
        b = _refit(Z, A, Y, groups, np.concatenate([rows_of[s] for s in pick]), blocks,
                   lam_rel, rank)
        bq.append(subspace_similarity(ref_Q, b.reply_loadings))
        bp.append(subspace_similarity(ref_P, b.partner_loadings))
    fq, fp = [], []
    for f in np.unique(fold_of_row[tr]):
        b = _refit(Z, A, Y, groups, tr[fold_of_row[tr] != f], blocks, lam_rel, rank)
        fq.append(subspace_similarity(ref_Q, b.reply_loadings))
        fp.append(subspace_similarity(ref_P, b.partner_loadings))
    return {"bootstrap_reply": float(np.mean(bq)), "bootstrap_partner": float(np.mean(bp)),
            "fold_reply_min": float(np.min(fq)), "fold_partner_min": float(np.min(fp))}


def h2_battery(Z: np.ndarray, A: np.ndarray, Y: np.ndarray, groups: np.ndarray,
               train_rows: np.ndarray, blocks: list[np.ndarray], reference: RRRResult,
               engine: RidgeEngine, donor_A: list[np.ndarray] | None, n_boot: int,
               fold_of_row: np.ndarray, seed: int) -> dict:
    """Requirements 1-5 for the reference fit (training rows only).

    Per-direction cosines, per-loading sign agreement and loading SDs are kept as
    descriptive output. The statistics that enter the verdict are the subspace
    similarities from :func:`stability_statistics`.
    """
    r = reference.rank
    ref_Q = reference.reply_loadings
    ref_P = reference.partner_loadings
    tr = np.flatnonzero(train_rows)
    sessions = np.unique(groups[tr])
    rows_of = {s: tr[groups[tr] == s] for s in sessions}
    rng = np.random.default_rng(seed)
    cosines, sign_keep, loads = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(sessions, size=len(sessions), replace=True)
        rows = np.concatenate([rows_of[s] for s in pick])
        b = _refit(Z, A, Y, groups, rows, blocks, reference.lam_rel, r)
        cos, j, sg = _match(ref_Q, b.reply_loadings)
        P = b.partner_loadings[:, j] * sg[None, :]
        cosines.append(cos)
        sign_keep.append(np.sign(P) == np.sign(ref_P))
        loads.append(P)
    fold_cos = []
    for f in np.unique(fold_of_row[tr]):
        rows = tr[fold_of_row[tr] != f]
        b = _refit(Z, A, Y, groups, rows, blocks, reference.lam_rel, r)
        fold_cos.append(_match(ref_Q, b.reply_loadings)[0])
    out = {
        "rank": r, "lambda_rel": reference.lam_rel,
        "strengths": reference.strengths.tolist(), "shares": reference.shares.tolist(),
        "concentration_leading_r": float(reference.shares[:r].sum()),
        "partner_loadings": ref_P.tolist(), "reply_loadings": ref_Q.tolist(),
        "bootstrap": {"n": n_boot,
                      "abs_cosine_mean": np.mean(cosines, axis=0).tolist(),
                      "sign_agreement_descriptive": np.mean(sign_keep, axis=0).tolist(),
                      "loading_sd": np.std(loads, axis=0).tolist()},
        "speaker_folds": {"abs_cosine": np.array(fold_cos).tolist(),
                          "abs_cosine_min": np.min(fold_cos, axis=0).tolist()},
        "stability": stability_statistics(Z, A, Y, groups, train_rows, blocks,
                                          reference.lam_rel, r, ref_Q, ref_P, fold_of_row,
                                          n_boot, seed + 1),
    }
    if donor_A is not None:
        obs = leading_strength(engine, A, reference.lam_rel)
        null = np.array([leading_strength(engine, Ad, reference.lam_rel) for Ad in donor_A])
        out["null"] = donor_p_value(obs, null)
    return out


def stability_null(Z, donor_A: list[np.ndarray], Y, groups, train_rows, blocks,
                   lam_rel: float, rank: int, fold_of_row: np.ndarray, n_boot: int,
                   seed: int) -> list[dict]:
    """The stability statistics recomputed with each donor partner matrix.

    Each donor gets its own reference fit (same λ and rank) and its own
    bootstrap and fold refits, so the null keeps everything fixed except the
    genuine pairing.
    """
    out = []
    tr = np.flatnonzero(train_rows)
    for k, Ad in enumerate(donor_A):
        ref = _refit(Z, Ad, Y, groups, tr, blocks, lam_rel, rank)
        out.append(stability_statistics(Z, Ad, Y, groups, train_rows, blocks, lam_rel, rank,
                                        ref.reply_loadings, ref.partner_loadings, fold_of_row,
                                        n_boot, seed + 10 + k))
    return out


STABILITY_KEYS = ("bootstrap_reply", "bootstrap_partner", "fold_reply_min", "fold_partner_min")


def h2_verdict(battery: dict, null_stats: list[dict] | None, alpha: float) -> dict:
    """The registered H2 null: no structure survives (2), (3) and (5) together.

    Calibrated rule: a stability statistic survives when it
    exceeds the (1 - alpha) quantile of the same statistic under the N1 donor
    refits; (5) survives when the leading-strength donor p-value is at most
    alpha. No fixed similarity threshold is used: in the synthetic study the
    speaker-fold statistic exceeded a conventional 0.85 congruence in about one
    null data set in ten.
    """
    if not null_stats or "null" not in battery:
        return {"status": "not_evaluated",
                "reason": "the calibrated rule needs the donor stability null and the "
                          "leading-strength null"}
    obs = battery["stability"]
    crit = {k: float(np.quantile([n[k] for n in null_stats], 1 - alpha))
            for k in STABILITY_KEYS}
    passes = {k: bool(obs[k] > crit[k]) for k in STABILITY_KEYS}
    strength_ok = battery["null"]["p"] <= alpha
    return {"status": "evaluated", "rule": "donor-calibrated subspace stability",
            "observed": obs, "null_quantile": crit, "passes": passes,
            "strength_p": battery["null"]["p"], "n_null": len(null_stats),
            "h2_null_rejected": bool(all(passes.values()) and strength_ok)}


def layer_consistency(partner_loadings: np.ndarray, probe_r2: np.ndarray) -> dict:
    """Spearman rank agreement between |partner loading| (leading direction) and probe R²."""
    from scipy.stats import spearmanr

    rho = spearmanr(np.abs(np.asarray(partner_loadings)[:, 0]), np.asarray(probe_r2))
    return {"spearman_rho": float(rho.statistic), "p_descriptive": float(rho.pvalue)}


__all__ = ["subspace_similarity", "stability_statistics", "h2_battery", "stability_null",
           "STABILITY_KEYS", "h2_verdict", "layer_consistency"]
