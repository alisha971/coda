"""Turn construction and pair construction.

The cases are the ones that decide what counts as a reply: an overlapped
backchannel, a backchannel in a pause, and interaction boundaries.
"""

import pandas as pd
import pytest

from src.data.pairs import build_pairs
from src.data.turns import build_ipus, build_turns


def words(*spans):
    return pd.DataFrame([(s, e, f"w{i}") for i, (s, e) in enumerate(spans)],
                        columns=["start", "end", "word"])


def test_ipus_split_only_at_silences_of_at_least_50ms():
    ipu = build_ipus(words((0.0, 0.2), (0.24, 0.5), (0.6, 0.9)))
    assert ipu[["start", "end", "n_words"]].to_numpy().tolist() == [[0.0, 0.5, 2], [0.6, 0.9, 1]]


def test_overlapped_backchannel_is_embedded_and_does_not_split_the_turn():
    a = words((0.0, 1.0), (1.2, 3.0))          # A: two IPUs
    b = words((2.0, 2.3))                       # B: "yeah" inside A's second IPU
    t = build_turns({"A": a, "B": b}, "I1", "S1")
    main = t[~t["embedded"]]
    assert main["speaker"].tolist() == ["A"]
    assert main["n_ipus"].iloc[0] == 2
    emb = t[t["embedded"]]
    assert emb["speaker"].tolist() == ["B"]
    assert emb["host_turn_id"].iloc[0] == "I1_T0000"


def test_backchannel_in_a_pause_takes_a_turn():
    a = words((0.0, 1.0), (2.0, 3.0))
    b = words((1.3, 1.6))                       # B speaks while A is silent
    t = build_turns({"A": a, "B": b}, "I1", "S1")
    assert t[~t["embedded"]]["speaker"].tolist() == ["A", "B", "A"]


def test_overlapping_takeover_is_a_transition():
    a = words((0.0, 2.0))
    b = words((1.8, 4.0))                       # starts in overlap, outlasts A
    t = build_turns({"A": a, "B": b}, "I1", "S1")
    assert t[~t["embedded"]]["speaker"].tolist() == ["A", "B"]


def test_turn_table_requires_two_speakers():
    with pytest.raises(ValueError):
        build_turns({"A": words((0, 1))}, "I1", "S1")


def _alternating(ik, sk, n):
    a = words(*[(2.0 * i, 2.0 * i + 0.8) for i in range(0, n, 2)])
    b = words(*[(2.0 * i, 2.0 * i + 0.8) for i in range(1, n, 2)])
    return build_turns({"A": a, "B": b}, ik, sk)


def test_pairs_never_cross_interactions_and_carry_self_previous():
    t = pd.concat([_alternating("I1", "S1", 4), _alternating("I2", "S1", 3)])
    p = build_pairs(t)
    assert len(p) == 3 + 2
    assert (p["partner_turn_id"].str[:2] == p["reply_turn_id"].str[:2]).all()
    assert (p["partner_speaker"] != p["replier"]).all()
    first = p[p["interaction_key"] == "I1"].iloc[0]
    assert pd.isna(first["self_previous_turn_id"])
    second = p[p["interaction_key"] == "I1"].iloc[1]
    assert second["self_previous_turn_id"] == "I1_T0000"


def test_n2_candidates_respect_the_minimum_turn_distance():
    t = _alternating("I1", "S1", 24)
    p = build_pairs(t, min_distance=10)
    # Partner turn 0 (speaker A): A holds turns 0,2,...,22; those >= 10 away are 10..22.
    assert p.iloc[0]["n2_candidates"] == 7
    assert (p["n2_candidates"] >= 0).all()


def test_embedded_turns_never_form_pairs():
    a = words((0.0, 1.0), (1.2, 3.0))
    b = words((2.0, 2.3), (3.5, 4.0))
    t = build_turns({"A": a, "B": b}, "I1", "S1")
    p = build_pairs(t)
    assert not p["reply_turn_id"].str.contains("_E").any()
    assert not p["partner_turn_id"].str.contains("_E").any()


def test_an_interaction_without_words_does_not_corrupt_the_table():
    """Regression: an empty interaction once made ``embedded`` an object column,
    and ``~`` on objects is bitwise-not (True -> -2), not logical negation."""
    empty = build_turns({"A": words(), "B": words()}, "I0", "S1")
    assert empty.empty and empty["embedded"].dtype == bool
    t = pd.concat([empty, _alternating("I1", "S1", 4)], ignore_index=True)
    assert t["duration"].dtype == float and t["n_words"].dtype.kind == "i"
    assert len(build_pairs(t)) == 3
    assert len(build_pairs(t.astype({"embedded": object}))) == 3
