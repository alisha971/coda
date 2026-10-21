"""Donors, donor p-values, Holm, and the session bootstrap (leakage register row 5)."""

import numpy as np
import pandas as pd
import pytest

from src.controls.donors import (
    DonorError,
    assert_donor_validity,
    draw_n1,
    draw_n2,
    load_donors,
    save_donors,
)
from src.data.pairs import build_pairs
from src.data.turns import build_turns
from src.evaluation.bootstrap import paired_partial_r2, resample_counts, self_test
from src.statistics.permutation import donor_p_value, holm


def corpus(n_sessions=12, turns_per=40, seed=0):
    rng = np.random.default_rng(seed)
    all_turns = []
    for s in range(n_sessions):
        sk, ik = f"V0_S{s:03d}", f"V0_S{s:03d}_I1"
        t, a, b = 0.0, [], []
        for k in range(turns_per):
            dur = float(rng.uniform(0.3, 3.0))
            nw = int(rng.integers(1, 6))
            spans = [(t + j * dur / nw, t + (j + 0.9) * dur / nw) for j in range(nw)]
            (a if k % 2 == 0 else b).extend(spans)
            t += dur + 0.3
        cols = ["start", "end", "word"]
        w = {f"V0_P{s}a": pd.DataFrame([(x, y, "w") for x, y in a], columns=cols),
             f"V0_P{s}b": pd.DataFrame([(x, y, "w") for x, y in b], columns=cols)}
        all_turns.append(build_turns(w, ik, sk))
    turns = pd.concat(all_turns, ignore_index=True)
    pairs = build_pairs(turns)
    sessions = sorted(turns["session_key"].unique())
    split = {sk: ("train" if i < 8 else "test") for i, sk in enumerate(sessions)}
    pairs["split"] = pairs["session_key"].map(split)
    pairs = pairs[pairs["n2_candidates"] > 0].reset_index(drop=True)
    return turns, pairs


def test_n1_donors_satisfy_every_registered_property():
    turns, pairs = corpus()
    d, rep = draw_n1(pairs, turns, n_draws=50, seed=1)
    assert_donor_validity(pairs, d, "N1_cross_dyad", turns)
    t = turns.set_index("turn_id")
    # Matching: where an exact word-count match was reported, word counts are equal.
    exact = rep.set_index("pair_id")["exact_word_match"]
    for i, row in pairs.iterrows():
        if exact[row["pair_id"]]:
            want = t.at[row["partner_turn_id"], "n_words"]
            assert (t.loc[d[i], "n_words"].to_numpy() == want).all()


def test_n2_donors_share_session_interaction_speaker_and_are_10_turns_away():
    turns, pairs = corpus()
    usable = set(turns["turn_id"])
    d = draw_n2(pairs, turns, usable, n_draws=50, seed=2)
    assert_donor_validity(pairs, d, "N2_within_dyad_distant", turns)


def test_donor_draws_are_reproducible_and_saved_exactly(tmp_path):
    turns, pairs = corpus()
    a, _ = draw_n1(pairs, turns, 20, seed=5)
    b, _ = draw_n1(pairs, turns, 20, seed=5)
    assert (a == b).all()
    h = save_donors(tmp_path / "n1.npz", a, pairs["pair_id"], "N1_cross_dyad", 5)
    back = load_donors(tmp_path / "n1.npz")
    assert (back["donors"] == a).all() and back["content_hash"] == h and back["seed"] == 5


@pytest.mark.parametrize("corrupt,match", [
    ("partner", "true partner"), ("other_split", "different split"),
    ("same_session_n1", "own session")])
def test_invalid_n1_assignments_are_rejected(corrupt, match):
    turns, pairs = corpus()
    d, _ = draw_n1(pairs, turns, 5, seed=3)
    i = 0
    if corrupt == "partner":
        d[i, 0] = pairs.loc[i, "partner_turn_id"]
    elif corrupt == "other_split":
        other = pairs.loc[pairs["split"] != pairs.loc[i, "split"], "partner_turn_id"].iloc[0]
        d[i, 0] = other
    else:
        same = pairs.loc[(pairs["session_key"] == pairs.loc[i, "session_key"])
                         & (pairs["partner_turn_id"] != pairs.loc[i, "reply_turn_id"])
                         & (pairs.index != i), "partner_turn_id"].iloc[3]
        d[i, 0] = same
    with pytest.raises(DonorError, match=match):
        assert_donor_validity(pairs, d, "N1_cross_dyad", turns)


def test_a_too_close_n2_donor_is_rejected():
    turns, pairs = corpus()
    d = draw_n2(pairs, turns, set(turns["turn_id"]), 5, seed=4)
    t = turns.set_index("turn_id")
    row = pairs.iloc[10]
    k = t.at[row["partner_turn_id"], "turn_index"]
    near = turns[(turns["interaction_key"] == row["interaction_key"])
                 & (turns["speaker"] == row["partner_speaker"])
                 & (turns["turn_index"].astype(float) == k + 2)]["turn_id"].iloc[0]
    d[10, 0] = near
    with pytest.raises(DonorError, match="turns away"):
        assert_donor_validity(pairs, d, "N2_within_dyad_distant", turns)


# -- p-values and Holm ------------------------------------------------------


def test_donor_p_value_is_the_registered_proportion():
    null = np.arange(100) / 100.0
    r = donor_p_value(0.955, null)
    assert r["p"] == pytest.approx(0.04)          # 0.96 .. 0.99 are >= 0.955
    assert donor_p_value(5.0, null)["reported"] == "< 0.01"


def test_donor_p_value_under_the_null_is_roughly_uniform():
    rng = np.random.default_rng(0)
    ps = [donor_p_value(rng.normal(), rng.normal(size=500))["p"] for _ in range(2000)]
    assert np.mean(np.array(ps) <= 0.05) == pytest.approx(0.05, abs=0.015)


def test_holm_matches_a_manual_computation():
    r = holm({"H2": 0.01, "H3": 0.04, "H4": 0.03})
    assert r["H2"]["p_holm"] == pytest.approx(0.03)
    assert r["H4"]["p_holm"] == pytest.approx(0.06)
    assert r["H3"]["p_holm"] == pytest.approx(0.06)
    assert r["H2"]["reject"] and not r["H3"]["reject"]


# -- bootstrap --------------------------------------------------------------


def test_bootstrap_self_test_passes():
    res = self_test(seed=0)
    assert res["passed"], res


def test_bootstrap_resamples_whole_sessions_only():
    c = resample_counts(30, 200, seed=1)
    assert (c.sum(axis=1) == 30).all() and np.allclose(c, np.round(c))


def test_paired_bootstrap_point_estimate_equals_the_direct_value():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(40), 50)
    ef = rng.exponential(0.9, size=(g.size, 8))
    er = ef + rng.exponential(0.1, size=ef.shape)
    blocks = [np.arange(0, 2), np.arange(2, 3), np.arange(3, 5), np.arange(5, 8)]
    out = paired_partial_r2(ef, er, g, blocks, n_boot=300, seed=2)
    direct = [1 - ef[:, b].sum() / er[:, b].sum() for b in blocks]
    np.testing.assert_allclose(out["block_point"], direct)
    lo, hi = out["intervals"][0.95]["aggregate"]
    assert lo < out["aggregate_point"] < hi


def test_interval_methods_agree_on_a_symmetric_case_and_bca_reduces_to_percentile():
    """The three intervals share resamples; with no bias and no skew, BCa is the
    percentile interval and all three are close."""
    from src.evaluation.bootstrap import interval_methods

    rng = np.random.default_rng(0)
    blocks = [np.arange(0, 2), np.arange(2, 3), np.arange(3, 5), np.arange(5, 8)]
    g = np.repeat(np.arange(60), 200)
    er = rng.gamma(2.0, 0.5, size=(g.size, 8))
    ef = er * (1 - 0.01) * rng.lognormal(0, 0.05, er.shape)
    out = interval_methods(ef, er, g, blocks, n_boot=1000, seed=1)
    for m in ("percentile", "bca", "bootstrap_t"):
        lo, hi = out[m]
        assert lo < out["point"] < hi
        assert hi - lo < 0.02
    assert abs(out["bca_acceleration"]) < 0.05
    width = np.diff(out["percentile"])[0]
    assert np.allclose(out["bca"], out["percentile"], atol=0.25 * width)


def test_bootstrap_t_standard_error_is_the_cluster_jackknife():
    from src.evaluation.bootstrap import interval_methods, session_sums

    rng = np.random.default_rng(2)
    blocks = [np.arange(0, 2), np.arange(2, 3), np.arange(3, 5), np.arange(5, 8)]
    g = np.repeat(np.arange(30), 50)
    er = rng.gamma(2.0, 0.5, size=(g.size, 8))
    ef = er * rng.uniform(0.95, 1.0, size=(30, 1))[g]
    out = interval_methods(ef, er, g, blocks, n_boot=200, seed=3)
    sf, _ = session_sums(ef, g)
    sr, _ = session_sums(er, g)
    tj = np.array([np.mean([1 - (sf.sum(0) - sf[i])[b].sum() / (sr.sum(0) - sr[i])[b].sum()
                            for b in blocks]) for i in range(30)])
    se = np.sqrt(29 / 30 * ((tj - tj.mean()) ** 2).sum())
    assert out["jackknife_se"] == pytest.approx(se)
