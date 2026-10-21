"""Standardisation (both modes) and within-block orthogonalisation (leakage register rows 2-3)."""

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_DIMENSIONS
from src.features.normstats import ProvenanceError, speaker_stats
from src.targets.normalise import (
    MODE_HELD_OUT,
    MODE_TRAIN,
    standardisation_check,
    standardise_rows,
)
from src.targets.orthogonalise import BlockOrthogonaliser

DIMS = list(TARGET_DIMENSIONS)


def prosody_table(n_turns=30, speakers=("A", "B"), sessions=("S1",), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in sessions:
        for spk in speakers:
            for k in range(n_turns):
                rows.append({"turn_id": f"{s}_{spk}_{k}", "speaker": f"{s}{spk}",
                             "session_key": s, "usable": True,
                             **{d: rng.normal(5 + ord(spk), 2) for d in DIMS}})
    return pd.DataFrame(rows)


def test_held_out_statistics_exclude_the_target_and_its_context_turns():
    """Perturbing an excluded turn leaves the value unchanged; an included one moves it."""
    p = prosody_table()
    target, ctx_self = "S1_A_5", "S1_A_4"
    excl = [[target, "S1_B_5", ctx_self]]
    base = standardise_rows([target], [MODE_HELD_OUT], excl, p, DIMS, None, 5)
    bumped = p.copy()
    bumped.loc[bumped["turn_id"] == ctx_self, DIMS] += 1000.0     # excluded context turn
    same = standardise_rows([target], [MODE_HELD_OUT], excl, bumped, DIMS, None, 5)
    np.testing.assert_allclose(same[DIMS].to_numpy(), base[DIMS].to_numpy())
    bumped2 = p.copy()
    bumped2.loc[bumped2["turn_id"] == "S1_A_20", DIMS] += 1000.0  # an ordinary turn
    moved = standardise_rows([target], [MODE_HELD_OUT], excl, bumped2, DIMS, None, 5)
    assert not np.allclose(moved[DIMS].to_numpy(), base[DIMS].to_numpy())


def test_held_out_value_matches_an_independent_computation():
    p = prosody_table()
    target = "S1_A_7"
    excl = [[target, "S1_B_7", "S1_A_6"]]
    got = standardise_rows([target], [MODE_HELD_OUT], excl, p, DIMS, None, 5)
    others = p[(p["speaker"] == "S1A") & ~p["turn_id"].isin([target, "S1_A_6"])]
    x = p.loc[p["turn_id"] == target, DIMS].to_numpy()[0]
    expected = (x - others[DIMS].mean().to_numpy()) / others[DIMS].std(ddof=1).to_numpy()
    np.testing.assert_allclose(got[DIMS].to_numpy()[0], expected, rtol=1e-9)
    assert got["n_stats_turns"].iloc[0] == 28


def test_held_out_statistics_never_cross_sessions():
    p = prosody_table(sessions=("S1", "S2"))
    p.loc[p["session_key"] == "S2", "speaker"] = p.loc[p["session_key"] == "S2", "speaker"] \
        .str.replace("S2", "S1")          # the same speaker label in another session
    one = standardise_rows(["S1_A_3"], [MODE_HELD_OUT], [[]], p, DIMS, None, 5)
    p2 = p.copy()
    p2.loc[p2["session_key"] == "S2", DIMS] += 500.0
    two = standardise_rows(["S1_A_3"], [MODE_HELD_OUT], [[]], p2, DIMS, None, 5)
    np.testing.assert_allclose(one[DIMS].to_numpy(), two[DIMS].to_numpy())


def test_too_few_turns_gives_a_reason_not_a_value():
    p = prosody_table(n_turns=4)
    r = standardise_rows(["S1_A_0"], [MODE_HELD_OUT], [["S1_A_1"]], p, DIMS, None, 10)
    assert r["standardisation_failure"].iloc[0] == "too_few_turns_for_stats"
    assert r[DIMS].isna().all(axis=None)


def test_training_mode_uses_training_statistics_only():
    p = prosody_table(speakers=("A", "B", "C"))
    train_ids = p.loc[p["speaker"].isin(["S1A", "S1B"]), "turn_id"]
    stats = speaker_stats(p, DIMS, train_ids, "h")
    stats.assert_training_only(train_ids)
    with pytest.raises(ProvenanceError):
        stats.assert_training_only(train_ids.iloc[:10])
    assert "S1C" not in stats.params["mean"].index
    r = standardise_rows(["S1_A_3"], [MODE_TRAIN], [[]], p, DIMS, stats, 5)
    a = p[p["speaker"] == "S1A"]
    expected = (a.loc[a["turn_id"] == "S1_A_3", DIMS].to_numpy()[0] - a[DIMS].mean().to_numpy()) \
        / a[DIMS].std(ddof=1).to_numpy()
    np.testing.assert_allclose(r[DIMS].to_numpy()[0], expected, rtol=1e-9)
    assert r["mode"].iloc[0] == MODE_TRAIN


def test_held_out_targets_come_out_near_but_not_exactly_zero_one():
    """Exactly 0/1 would indicate that test rows fed their own statistics."""
    p = prosody_table(n_turns=60)
    ids = p.loc[p["speaker"] == "S1A", "turn_id"].tolist()
    r = standardise_rows(ids, [MODE_HELD_OUT] * len(ids), [[]] * len(ids), p, DIMS, None, 5)
    chk = standardisation_check(r[DIMS].to_numpy())
    assert not chk["exactly_zero_one"]
    assert np.all(np.abs(chk["mean"]) < 0.2) and np.all(np.abs(np.array(chk["sd"]) - 1) < 0.2)
    leaky = (p.loc[p["speaker"] == "S1A", DIMS] - p.loc[p["speaker"] == "S1A", DIMS].mean()) \
        / p.loc[p["speaker"] == "S1A", DIMS].std(ddof=1)
    assert standardisation_check(leaky.to_numpy())["exactly_zero_one"]


# -- orthogonalisation ------------------------------------------------------


def correlated_targets(n, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(n, 8)) + shift
    Y[:, 1] += 0.8 * Y[:, 0]          # pitch level and span correlated
    Y[:, 4] += 0.6 * Y[:, 3]          # energy pair correlated
    Y[:, 6] += 0.5 * Y[:, 5]
    Y[:, 7] -= 0.4 * Y[:, 6]
    return Y


def test_orthogonalised_training_targets_have_identity_covariance_per_block():
    Y = correlated_targets(5000)
    o = BlockOrthogonaliser().fit(Y, range(len(Y)), "h")
    Z = o.transform(Y)
    for idx in o.index.values():
        c = np.cov(Z[:, idx], rowvar=False)
        np.testing.assert_allclose(np.atleast_2d(c), np.eye(len(idx)), atol=1e-8)


def test_zca_cor_keeps_each_output_closest_to_its_own_input():
    Y = correlated_targets(5000, seed=1)
    Z = BlockOrthogonaliser().fit(Y, range(len(Y)), "h").transform(Y)
    c = np.corrcoef(np.hstack([Y, Z]), rowvar=False)[:8, 8:]
    o = BlockOrthogonaliser()
    for idx in o.index.values():
        for a in idx:
            assert np.argmax(np.abs(c[idx, a])) == idx.index(a)


def test_transform_fitted_on_train_is_reused_unchanged_on_held_out_rows():
    Ytr, Yte = correlated_targets(4000, 2), correlated_targets(1000, 3, shift=0.3)
    o = BlockOrthogonaliser().fit(Ytr, range(4000), "h")
    params_before = {b: p["W"].copy() for b, p in o.fitted.params.items()}
    Zte = o.transform(Yte)
    for b, p in o.fitted.params.items():
        np.testing.assert_array_equal(p["W"], params_before[b])
    assert abs(Zte.mean()) > 0.05      # the held-out shift is not absorbed: nothing refitted
