"""Splits and the per-pair information firewall (leakage register rows 1, 4).

Several cases are built to leak under a naive implementation, so that the
assertion, not luck, is what keeps the partitions clean.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import (
    LeakageError,
    PairFirewall,
    build_analysis_set,
    check_temporal_order,
    exclusion_table,
)
from src.data.pairs import build_pairs
from src.data.splits import (
    SplitLeakage,
    assert_disjoint,
    make_split,
    manifest_hash,
    near_duplicate_links,
    split_report,
)
from src.data.turns import build_turns

FR = {"train": 0.7, "val": 0.1, "test": 0.2}


def sessions_table(n: int, shared: list[tuple[int, int]] = ()) -> pd.DataFrame:
    """n sessions with two speakers each; ``shared`` makes session j reuse session i's speaker."""
    rows = []
    for i in range(n):
        rows += [(f"S{i:03d}", f"P{i:03d}a"), (f"S{i:03d}", f"P{i:03d}b")]
    for i, j in shared:
        rows = [(s, f"P{i:03d}a") if (s == f"S{j:03d}" and p.endswith("a")) else (s, p)
                for s, p in rows]
    return pd.DataFrame(rows, columns=["session_key", "participant_key"])


# -- splits -----------------------------------------------------------------


def test_split_is_speaker_disjoint_and_covers_every_session():
    s = sessions_table(100)
    m = make_split(s, FR, seed=1)
    assert set(m["session_key"]) == set(s["session_key"])
    counts = m["split"].value_counts()
    assert counts["train"] == 70 and counts["val"] == 10 and counts["test"] == 20


def test_sessions_sharing_a_speaker_are_kept_together():
    """Identifier reuse must not let one person sit in two partitions."""
    s = sessions_table(50, shared=[(3, 40), (7, 8)])
    for seed in range(20):
        m = make_split(s, FR, seed=seed).set_index("session_key")["split"]
        assert m["S003"] == m["S040"] and m["S007"] == m["S008"]


def test_a_naive_per_session_split_would_leak_and_is_caught():
    s = sessions_table(10, shared=[(0, 1)])
    naive = pd.DataFrame({"session_key": [f"S{i:03d}" for i in range(10)],
                          "split": ["train", "test"] + ["train"] * 8})
    with pytest.raises(SplitLeakage, match="participants"):
        assert_disjoint(naive, s)


def test_near_duplicate_sessions_go_to_one_partition():
    s = sessions_table(30)
    text = {f"S{i:03d}": " ".join(f"w{i}_{k}" for k in range(60)) for i in range(30)}
    text["S029"] = text["S005"]                     # a duplicated release
    links = near_duplicate_links(text, k=5, threshold=0.5)
    assert ("S005", "S029") in {(a, b) for a, b, _ in links}
    for seed in range(10):
        m = make_split(s, FR, seed=seed, near_duplicates=links).set_index("session_key")
        assert m.loc["S005", "split"] == m.loc["S029", "split"]


def test_split_is_reproducible_and_seed_dependent():
    s = sessions_table(60)
    a, b, c = make_split(s, FR, 7), make_split(s, FR, 7), make_split(s, FR, 8)
    assert manifest_hash(a) == manifest_hash(b)
    assert manifest_hash(a) != manifest_hash(c)


def test_dyad_disjoint_contrast_reports_its_shared_speakers():
    """With one session per speaker the contrast shares no speaker."""
    s = sessions_table(40)
    m = make_split(s, FR, 3, scheme="dyad_disjoint")
    assert split_report(m, s)["speakers_shared_across_partitions"] == 0
    s2 = sessions_table(40, shared=[(1, 2)])
    shared = [split_report(make_split(s2, FR, seed, scheme="dyad_disjoint"), s2)
              ["speakers_shared_across_partitions"] for seed in range(30)]
    assert max(shared) == 1        # the contrast can separate a reused speaker; primary cannot


def test_duplicate_session_rows_in_a_manifest_are_rejected():
    s = sessions_table(5)
    m = pd.DataFrame({"session_key": ["S000", "S000", "S001", "S002", "S003", "S004"],
                      "split": ["train", "test", "train", "train", "val", "test"]})
    with pytest.raises(SplitLeakage, match="more than once"):
        assert_disjoint(m, s)


# -- firewall ---------------------------------------------------------------


def _words(*spans):
    return pd.DataFrame([(s, e, f"w{i}") for i, (s, e) in enumerate(spans)],
                        columns=["start", "end", "word"])


@pytest.fixture
def conversation():
    a = _words(*[(2.0 * i, 2.0 * i + 0.8) for i in range(0, 30, 2)])
    b = _words(*[(2.0 * i, 2.0 * i + 0.8) for i in range(1, 30, 2)])
    turns = build_turns({"V0_P1": a, "V0_P2": b}, "V0_S1_I1", "V0_S1")
    turns["delivery"] = np.arange(len(turns), dtype=float)
    turns["duration_true"] = turns["duration"]
    pairs = build_pairs(turns)
    return turns, pairs


def test_firewall_refuses_the_pairs_own_reply_and_serves_it_as_context(conversation):
    """Leakage register row 1, both directions."""
    turns, pairs = conversation
    fw = PairFirewall(turns, pairs)
    first, second = pairs.iloc[0], pairs.iloc[1]
    assert second["partner_turn_id"] == first["reply_turn_id"]
    for field in ("duration", "audio", "ssl", "delivery", "mean_log_f0"):
        with pytest.raises(LeakageError):
            fw.request(first["pair_id"], "reply", field)
    # The same turn, as the partner of the next pair, is served normally.
    v = fw.request(second["pair_id"], "partner", "duration")
    assert v == turns.set_index("turn_id").at[first["reply_turn_id"], "duration"]


def test_reply_words_are_permitted_because_the_design_conditions_on_them(conversation):
    turns, pairs = conversation
    fw = PairFirewall(turns, pairs)
    assert isinstance(fw.request(pairs.iloc[0]["pair_id"], "reply", "text"), str)
    assert fw.request(pairs.iloc[0]["pair_id"], "reply", "n_words") == 1


def test_vectorised_gather_applies_the_same_rule(conversation):
    turns, pairs = conversation
    fw = PairFirewall(turns, pairs)
    with pytest.raises(LeakageError):
        fw.gather(pairs["pair_id"], "reply", ["n_words", "duration"])
    got = fw.gather(pairs["pair_id"], "partner", ["delivery"])
    assert len(got) == len(pairs)


def test_a_pairs_own_reply_can_never_be_its_donor(conversation):
    turns, pairs = conversation
    fw = PairFirewall(turns, pairs)
    p = pairs.iloc[3]
    with pytest.raises(LeakageError):
        fw.request(p["pair_id"], "donor", "delivery", donor_turn=p["reply_turn_id"])
    with pytest.raises(LeakageError):
        fw.gather(pairs["pair_id"], "donor", ["delivery"], donor_turns=pairs["reply_turn_id"])


def test_temporal_order_violation_is_detected(conversation):
    turns, pairs = conversation
    check_temporal_order(pairs, turns)
    swapped = pairs.copy()
    swapped.loc[0, ["partner_turn_id", "reply_turn_id"]] = \
        swapped.loc[0, ["reply_turn_id", "partner_turn_id"]].to_numpy()
    with pytest.raises(LeakageError, match="partner"):
        check_temporal_order(swapped, turns)


def test_duplicate_pair_ids_are_rejected(conversation):
    turns, pairs = conversation
    with pytest.raises(ValueError):
        PairFirewall(turns, pd.concat([pairs, pairs.iloc[:1]]))


def test_analysis_set_reasons_are_complete_and_add_up(conversation):
    turns, pairs = conversation
    tids = turns.loc[~turns["embedded"], "turn_id"]
    usable = pd.Series(True, index=tids)
    usable.iloc[5] = False
    prosody = pd.DataFrame({"turn_id": tids.to_numpy(), "usable": usable.to_numpy()})
    manifest = pd.DataFrame({"session_key": ["V0_S1"], "split": ["train"]})
    a = build_analysis_set(pairs, turns, prosody, manifest)
    assert a["included"].sum() + a["exclusion"].notna().sum() == len(pairs)
    assert set(a.loc[~a["included"], "exclusion"]) <= {
        "reply_not_usable", "partner_not_usable", "no_self_previous",
        "self_previous_not_usable", "no_n2_candidate"}
    assert a.loc[a["reply_turn_position"] == 1, "exclusion"].iloc[0] == "no_self_previous"
    tab = exclusion_table(a)
    assert tab.loc["All", "All"] == len(pairs)


def test_simultaneous_onset_is_excluded_with_a_reason_and_reversals_still_raise():
    """Regression (notebook 10 on the pilot raised 'a partner turn does not start before its
    reply'): two speakers whose turns start at the same timestamp form a pair without temporal
    precedence. It is excluded as 'simultaneous_onset'; a genuine reversal still raises."""
    from src.data.dataset import build_analysis_set, simultaneous_onset

    def w(*spans):
        return pd.DataFrame([{"start": s, "end": e, "word": f"w{s}"} for s, e in spans])

    turns = build_turns({"A": w((0.0, 1.0), (3.0, 3.5), (5.0, 5.5)),
                         "B": w((0.0, 2.0), (4.0, 4.5), (6.0, 6.5))}, "S1_I1", "S1")
    pairs = build_pairs(turns, min_distance=1)
    ties = simultaneous_onset(pairs, turns)
    # both turns start at 0.0: the tie removes the pair they form (partner, reply) and the
    # next pair, where they are (self-previous, partner)
    assert ties.tolist() == [True, True, False, False, False]
    prosody = pd.DataFrame({"turn_id": turns["turn_id"], "usable": True})
    split = pd.DataFrame({"session_key": ["S1"], "split": ["train"]})
    out = build_analysis_set(pairs, turns, prosody, split)   # no raise
    assert (out.loc[ties.to_numpy(), "exclusion"] == "simultaneous_onset").all()
    assert not out.loc[ties.to_numpy(), "included"].any()
    swapped = pairs[~ties].copy().reset_index(drop=True)
    swapped.loc[0, ["partner_turn_id", "reply_turn_id"]] = \
        swapped.loc[0, ["reply_turn_id", "partner_turn_id"]].to_numpy()
    with pytest.raises(LeakageError, match="partner"):
        build_analysis_set(swapped, turns, prosody, split)


def test_pairs_whose_distant_partner_turns_are_all_unusable_are_excluded_before_n2_draws():
    """Regression (notebook 12 on the pilot: 'no usable N2 donor'): inclusion counted N2
    candidates without asking whether their delivery was usable; the donor draw does ask.
    Both now use one rule, so every included pair has a drawable N2 donor."""
    from src.controls.donors import draw_n2
    from src.data.dataset import build_analysis_set

    def w(*starts):
        return pd.DataFrame([{"start": s, "end": s + 0.5, "word": f"w{s}"} for s in starts])

    n = 30                                    # strictly alternating turns A, B, A, B, ...
    turns = build_turns({"A": w(*[float(2 * i) for i in range(n // 2)]),
                         "B": w(*[float(2 * i + 1) for i in range(n // 2)])}, "S1_I1", "S1")
    pairs = build_pairs(turns, min_distance=10)
    # every turn usable except speaker A's turns from index 12 on: pairs whose partner is an
    # early A turn then have distant A turns, but none usable
    t = turns[~turns["embedded"].astype(bool)]
    usable = ~((t["speaker"] == "A") & (t["turn_index"].astype(int) >= 12))
    prosody = pd.DataFrame({"turn_id": t["turn_id"], "usable": usable.to_numpy()})
    split = pd.DataFrame({"session_key": ["S1"], "split": ["train"]})
    out = build_analysis_set(pairs, turns, prosody, split)
    starved = out["exclusion"] == "no_n2_candidate"
    structural = out["n2_candidates"].astype(int) > 0
    assert (starved & structural).any()       # the case the old rule let through
    inc = out[out["included"]]
    assert len(inc) > 0
    usable_ids = set(prosody.loc[prosody["usable"], "turn_id"])
    draw_n2(inc, turns, usable_ids, n_draws=5, seed=1)     # no DonorError
