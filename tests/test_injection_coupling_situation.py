"""Positive control, the H2 battery, mixed effects and the situation analysis."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.coupling_map import (
    STABILITY_KEYS,
    h2_battery,
    h2_verdict,
    stability_null,
    subspace_similarity,
)
from src.analysis.situation import (
    LeakageError,
    classifier_inputs,
    cohen_kappa,
    confirmatory_strata,
    g4_verdict,
    hand_label_sample,
    interrogative,
    kappa_per_category,
    label_pairs,
)
from src.controls.injection import (
    coupling_matrix,
    injected_targets,
    mde_from_power,
    rejection_rate,
    session_block_residuals,
)
from src.evaluation.metrics import block_partial_r2
from src.models.ridge import RidgeEngine, ZCache, block_col_groups
from src.models.rrr import fit_rrr
from src.settings import Settings

BLOCKS = block_col_groups([2, 1, 2, 3])


def base_data(n_sessions=80, per=50, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_sessions), per)
    n = g.size
    Z = rng.normal(size=(n, 10))
    A = rng.normal(size=(n, 8)) + 0.2 * Z[:, :8]
    fhat = Z @ rng.normal(size=(10, 8)) * 0.2
    eps = rng.normal(size=(n, 8)) + rng.normal(size=(n_sessions, 8))[g] * 0.4
    test = g >= int(0.8 * n_sessions)
    return Z, A, fhat, eps, g, ~test, test


# -- injection --------------------------------------------------------------


def test_cross_map_is_a_permutation_across_blocks():
    G = coupling_matrix("cross_dimension_variant",
                        Settings().injection["cross_variant_permutation"])
    assert (G.sum(0) == 1).all() and (G.sum(1) == 1).all()
    assert np.trace(G) == 0
    with pytest.raises(ValueError, match="own block"):
        coupling_matrix("cross_dimension_variant", {"mean_log_f0": "log_f0_range_p95_p5"})


def test_session_block_residuals_keep_sessions_from_one_source():
    rng = np.random.default_rng(1)
    pool = np.arange(200, dtype=float)[:, None] * np.ones((1, 8))
    pool_g = np.repeat(np.arange(20), 10)
    g = np.repeat(np.arange(5), 7)
    eps = session_block_residuals(pool, pool_g, g, rng)
    for s in range(5):
        src = np.unique(pool_g[eps[g == s, 0].astype(int)])
        assert len(src) == 1


@pytest.mark.parametrize("target", [0.01, 0.05])
def test_injected_effect_has_roughly_the_target_partial_r2(target):
    Z, A, fhat, eps, g, fit, test = base_data(n_sessions=120)
    Y, beta = injected_targets(fhat, A, Z, eps[fit], g[fit], g, fit, target, np.eye(8), BLOCKS,
                               n_replicates=4, seed=2)
    zc = ZCache(Z, g, fit, test)
    eng = RidgeEngine(zc, Y, block_col_groups([2, 1, 2, 3], 4))
    full, red = eng.run(A), eng.run(None)
    pr2 = [np.mean(block_partial_r2(Y[test][:, k * 8:(k + 1) * 8],
                                    full.pred_eval[:, k * 8:(k + 1) * 8],
                                    red.pred_eval[:, k * 8:(k + 1) * 8], BLOCKS))
           for k in range(4)]
    assert np.mean(pr2) == pytest.approx(target, abs=0.5 * target + 0.004)


def test_beta_zero_gives_no_dependence_on_a():
    Z, A, fhat, eps, g, fit, test = base_data()
    Y, beta = injected_targets(fhat, A, Z, eps[fit], g[fit], g, fit, 0.0, np.eye(8), BLOCKS,
                               2, 3)
    assert (beta == 0).all()
    np.testing.assert_allclose(Y[:, :8] - fhat, Y[:, :8] - fhat)  # f̂ + ε only
    assert abs(np.corrcoef((Y[:, :8] - fhat)[:, 0], A[:, 0])[0, 1]) < 0.05


def test_mde_interpolates_and_reports_none_when_power_is_never_reached():
    assert mde_from_power([0.01, 0.02, 0.05], [0.2, 0.6, 1.0], 0.8) == pytest.approx(0.035)
    assert mde_from_power([0.01, 0.02], [0.1, 0.3], 0.8) is None


def test_rejection_rate_carries_a_binomial_interval():
    r = rejection_rate(np.r_[np.full(10, 0.01), np.full(190, 0.5)], 0.05)
    assert r["rate"] == pytest.approx(0.05)
    assert r["ci95"][0] < 0.05 < r["ci95"][1]


# -- H2 battery -------------------------------------------------------------


def test_h2_battery_is_stable_for_a_planted_direction():
    rng = np.random.default_rng(5)
    Z, A, fhat, eps, g, fit, test = base_data(n_sessions=100)
    u, v = rng.normal(size=(8, 1)), rng.normal(size=(1, 8))
    Y = fhat + 0.5 * A @ (u @ v) / np.linalg.norm(u @ v) * 3 + eps
    zc = ZCache(Z, g, fit, test)
    eng = RidgeEngine(zc, Y, BLOCKS)
    ref = fit_rrr(eng, A, BLOCKS)
    folds = np.full(len(g), -1)
    folds[zc.fit_idx] = zc.fold
    donors = [A[rng.permutation(len(A))] for _ in range(20)]
    bat = h2_battery(Z, A, Y, g, fit, BLOCKS, ref, eng, donors, n_boot=20,
                     fold_of_row=folds, seed=1)
    assert bat["rank"] <= 2
    assert bat["bootstrap"]["abs_cosine_mean"][0] > 0.95
    assert min(bat["speaker_folds"]["abs_cosine_min"]) > 0.9 or bat["rank"] == 1
    assert bat["null"]["p"] == 0.0
    assert h2_verdict(bat, None, 0.05)["status"] == "not_evaluated"
    assert bat["stability"]["bootstrap_reply"] > 0.9
    nulls = stability_null(Z, donors[:9], Y, g, fit, BLOCKS, ref.lam_rel, ref.rank, folds,
                           n_boot=5, seed=2)
    v = h2_verdict(bat, nulls, 0.05)
    assert v["status"] == "evaluated" and v["h2_null_rejected"]


def test_subspace_similarity_ignores_rotation_inside_a_stable_plane():
    """Counterexample to direction matching: two equally strong directions swap order."""
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(8, 2)))
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    assert subspace_similarity(Q, Q @ R) == pytest.approx(1.0)
    assert subspace_similarity(Q, Q[:, ::-1] * np.array([1, -1])) == pytest.approx(1.0)
    perp = np.linalg.qr(np.hstack([Q, rng.normal(size=(8, 2))]))[0][:, 2:]
    assert subspace_similarity(Q, perp) == pytest.approx(0.0, abs=1e-12)


def test_near_zero_partner_loading_does_not_break_the_stability_statistic():
    """Counterexample to per-loading sign agreement: a loading near zero flips sign
    at random across refits, so sign agreement is about 0.5 while the direction
    is essentially unchanged."""
    rng = np.random.default_rng(1)
    p = np.array([1.0, 0.8, -0.6, 0.5, 0.4, -0.3, 0.2, 0.0])[:, None]
    signs, sims = [], []
    for _ in range(200):
        q = p + 0.02 * rng.normal(size=p.shape)
        signs.append(np.sign(q[-1, 0]) == 1.0)
        sims.append(subspace_similarity(p, q))
    assert 0.3 < np.mean(signs) < 0.7          # the old rule would call this unstable
    assert min(sims) > 0.99


def test_h2_null_case_does_not_survive():
    rng = np.random.default_rng(7)
    Z, A, fhat, eps, g, fit, test = base_data(n_sessions=60)
    Y = fhat + eps                                       # no partner coupling
    zc = ZCache(Z, g, fit, test)
    eng = RidgeEngine(zc, Y, BLOCKS)
    ref = fit_rrr(eng, A, BLOCKS)
    folds = np.full(len(g), -1)
    folds[zc.fit_idx] = zc.fold
    donors = [A[rng.permutation(len(A))] for _ in range(19)]
    bat = h2_battery(Z, A, Y, g, fit, BLOCKS, ref, eng, donors, n_boot=8,
                     fold_of_row=folds, seed=3)
    nulls = stability_null(Z, donors, Y, g, fit, BLOCKS, ref.lam_rel, ref.rank, folds,
                           n_boot=8, seed=4)
    v = h2_verdict(bat, nulls, 0.05)
    assert set(v["passes"]) == set(STABILITY_KEYS)
    assert not v["h2_null_rejected"]


# -- mixed effects ----------------------------------------------------------


def test_mixed_model_recovers_a_partner_effect_and_its_interaction():
    pytest.importorskip("statsmodels")
    from src.statistics.mixed_effects import aggregate_h3, interaction_model, partner_model

    rng = np.random.default_rng(6)
    n_s, per = 40, 60
    sess = np.repeat(np.arange(n_s), per)
    spk = sess * 2 + rng.integers(0, 2, sess.size)
    fn = rng.choice(["a", "b"], sess.size)
    a = rng.normal(size=sess.size)
    z = rng.normal(size=sess.size)
    slope = np.where(fn == "b", 0.6, 0.1)
    y = slope * a + 0.3 * z + rng.normal(0, 0.5, n_s)[sess] + rng.normal(0, 0.3, 2 * n_s)[spk] \
        + rng.normal(size=sess.size)
    df = pd.DataFrame({"y": y, "a": a, "z": z, "function": fn,
                       "session_key": [f"s{s}" for s in sess], "replier": [f"p{s}" for s in spk]})
    pm = partner_model(df, "y", ["a"], ["z"])
    assert pm["wald_partner_block"]["p"] < 1e-6
    im = interaction_model(df, "y", ["a"], ["z"])
    assert im["wald_interaction"]["p"] < 1e-6
    assert im["interaction_coefficients"]["a_x_b"] == pytest.approx(0.5, abs=0.15)
    assert aggregate_h3({"d1": 0.01, "d2": 0.5}, "holm_min_over_dimensions") == pytest.approx(0.02)


# -- situation --------------------------------------------------------------


@pytest.mark.parametrize("text,q", [("where did you go", True), ("Do you like it", True),
                                    ("yeah that's right.", False), ("is it?", True),
                                    ("I think so", False), ("", False)])
def test_interrogative_rule(text, q):
    assert interrogative(text) is q


def test_classifier_sees_text_only():
    turns = pd.DataFrame({"turn_id": ["a", "b"], "text": ["what time is it", "noon"]})
    pairs = pd.DataFrame({"pair_id": ["p"], "partner_turn_id": ["a"], "reply_turn_id": ["b"]})
    inp = classifier_inputs(pairs, turns)
    assert list(inp.columns) == ["pair_id", "partner_text", "reply_text"]
    with pytest.raises(LeakageError, match="text only"):
        label_pairs(inp.assign(mean_log_f0=1.0), lambda p, r: "other")
    out = label_pairs(inp, None)
    assert out["function"].iloc[0] == "question_to_answer"


def test_without_a_chosen_classifier_non_questions_stay_unlabelled():
    inp = pd.DataFrame({"pair_id": ["p"], "partner_text": ["I went home"],
                        "reply_text": ["oh nice"]})
    assert label_pairs(inp, None)["function"].iloc[0] == "unlabelled_classifier_blocked"


def test_kappa_matches_sklearn_and_g4_rules():
    from sklearn.metrics import cohen_kappa_score

    rng = np.random.default_rng(7)
    a = rng.choice(["x", "y", "z"], 300)
    b = np.where(rng.random(300) < 0.7, a, rng.choice(["x", "y", "z"], 300))
    assert cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))
    k = {"question_to_answer": 0.8, "assessment_opinion_to_response": 0.65,
         "narrative_inform_to_continuation": 0.4, "backchannel_adjacent": 0.59}
    assert g4_verdict(k)["status"] == "PASS"
    assert g4_verdict({**k, "assessment_opinion_to_response": 0.5})["status"] == "FAIL"
    labels = pd.Series(["question_to_answer"] * 250 + ["assessment_opinion_to_response"] * 150)
    s = confirmatory_strata(labels, k).set_index("category")
    assert s.loc["question_to_answer", "role"] == "confirmatory"
    assert s.loc["assessment_opinion_to_response", "why"] == "fewer than 200 pairs"
    hand = pd.Series(["question_to_answer", "other"])
    assert set(kappa_per_category(hand, hand)) == set(k)


def test_hand_label_sample_is_seeded_and_blind():
    inp = pd.DataFrame({"pair_id": [f"p{i}" for i in range(1000)], "partner_text": "a",
                        "reply_text": "b"})
    s1, s2 = hand_label_sample(inp, 300, 1), hand_label_sample(inp, 300, 1)
    assert (s1["pair_id"] == s2["pair_id"]).all() and len(s1) == 300
    assert "function" not in s1.columns


def test_a_failed_mixed_model_fit_is_recorded_not_raised(monkeypatch):
    """Regression: a singular fit used to abort the whole J1 run."""
    pytest.importorskip("statsmodels")
    from src.statistics import mixed_effects as me

    def boom(*a, **k):
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(me, "_fit", boom)
    df = pd.DataFrame({"y": [1.0, 2.0], "a": [0.0, 1.0], "session_key": ["s", "s"],
                       "replier": ["p", "p"]})
    out = me.partner_model(df, "y", ["a"], [])
    assert out["fit_failed"] and "Singular" in out["error"]
    assert me.aggregate_h3({"d1": 0.01, "d2": None}, "holm_min_over_dimensions") is None


def _lag1(x, g):
    same = g[1:] == g[:-1]
    return float(np.corrcoef(x[1:][same], x[:-1][same])[0, 1])


def test_ordered_session_block_resampling_keeps_serial_dependence():
    """The design asks for residuals 'preserving their covariance'. Row-wise resampling inside a
    session keeps the cross-dimension covariance but destroys lag-1 dependence; the ordered
    scheme keeps both."""
    from src.controls.injection import session_block_residuals

    rng = np.random.default_rng(0)
    n_s, L, rho = 40, 200, 0.7
    e = rng.normal(size=(n_s, L, 2))
    x = np.empty_like(e)
    x[:, 0] = e[:, 0]
    for t in range(1, L):
        x[:, t] = rho * x[:, t - 1] + np.sqrt(1 - rho ** 2) * e[:, t]
    x[..., 1] += 0.8 * x[..., 0]                         # cross-dimension covariance
    pool = x.reshape(-1, 2)
    g = np.repeat(np.arange(n_s), L)
    target_g = np.repeat(np.arange(30), 150)
    rows = session_block_residuals(pool, g, target_g, np.random.default_rng(1))
    ordered = session_block_residuals(pool, g, target_g, np.random.default_rng(1), ordered=True)
    assert abs(_lag1(rows[:, 0], target_g)) < 0.1
    assert _lag1(ordered[:, 0], target_g) == pytest.approx(rho, abs=0.08)
    c_pool = np.corrcoef(pool.T)[0, 1]
    for out in (rows, ordered):
        assert np.corrcoef(out.T)[0, 1] == pytest.approx(c_pool, abs=0.05)


def test_unknown_residual_resampling_is_rejected():
    from src.experiments.layer1 import j3_level

    with pytest.raises(ValueError, match="residual resampling"):
        j3_level(None, None, {"rows": None}, 0.0, "identity_matched_dimensions", {}, 1, {},
                 0.05, 0, resampling="rows_shuffled")
