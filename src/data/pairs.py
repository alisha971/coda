"""Ordered turn pairs: the unit of analysis.

A pair is a partner turn followed by the reply turn: two consecutive
non-embedded turns by different speakers in the same interaction. Pairs never
cross an interaction boundary, because the time between interactions was not
released and consecutive interactions are not recoverable in order.

Each pair also carries:

- the **self-previous turn**, the replier's own turn immediately before the
  partner turn, used by the ``+self_previous`` condition for H4;
- the **turn position** of the reply within its interaction;
- the **gap** from partner end to reply start (negative when the reply begins
  in overlap);
- the number of **N2 candidates**: turns by the partner speaker in the same
  interaction at least ``min_distance`` turns from the partner turn. This is
  an availability count for the funnel; donors are drawn in
  ``src/controls/donors.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PAIR_COLUMNS = [
    "pair_id", "session_key", "interaction_key", "partner_turn_id", "reply_turn_id",
    "self_previous_turn_id", "partner_speaker", "replier", "reply_turn_position",
    "turns_in_interaction", "gap_s", "n2_candidates",
]


def _n2_counts(n: int, min_distance: int) -> np.ndarray:
    """For partner position j = 0..n-2 in a strictly alternating sequence, the
    number of same-speaker positions at least ``min_distance`` away."""
    j = np.arange(n - 1)
    idx = np.arange(n)
    same = (idx[None, :] - j[:, None]) % 2 == 0
    far = np.abs(idx[None, :] - j[:, None]) >= min_distance
    return (same & far).sum(axis=1)


def usable_n2_counts(pairs: pd.DataFrame, turns: pd.DataFrame, usable_turns,
                     min_distance: int = 10) -> pd.Series:
    """Per pair, the number of N2 donors that can actually be drawn: turns by the
    partner speaker in the same interaction, at least ``min_distance`` turns from
    the partner turn, whose delivery is usable.

    This is the eligibility rule of ``controls.donors.draw_n2``; the analysis set
    uses it so that every included pair has a donor ("N2 available"). The
    structural count ``n2_candidates`` ignores usability and is kept only as a
    description of the conversation.
    """
    usable = set(usable_turns)
    main = turns[~turns["embedded"].astype(bool)]
    pool = {}
    for key, g in main.groupby(["interaction_key", "speaker"]):
        ok = g["turn_id"].isin(usable).to_numpy()
        pool[key] = np.sort(g["turn_index"].astype(int).to_numpy()[ok])
    k_of = main.set_index("turn_id")["turn_index"]
    out = np.zeros(len(pairs), dtype=int)
    for i, (ik, spk, ptid) in enumerate(zip(pairs["interaction_key"], pairs["partner_speaker"],
                                           pairs["partner_turn_id"], strict=True)):
        idx = pool.get((ik, spk))
        if idx is None or ptid not in k_of.index or pd.isna(k_of[ptid]):
            continue
        k = int(k_of[ptid])
        out[i] = int(np.sum(np.abs(idx - k) >= min_distance))
    return pd.Series(out, index=pairs.index, name="usable_n2_donors")


def build_pairs(turns: pd.DataFrame, min_distance: int = 10) -> pd.DataFrame:
    """All candidate pairs from a turns table covering one or more interactions."""
    embedded = turns["embedded"].astype(bool)  # never rely on the incoming dtype
    main = turns[~embedded].sort_values(["interaction_key", "turn_index"])
    frames = []
    for ik, t in main.groupby("interaction_key", sort=True):
        n = len(t)
        if n < 2:
            continue
        spk = t["speaker"].to_numpy()
        if (spk[1:] == spk[:-1]).any():
            raise AssertionError(f"{ik}: consecutive non-embedded turns share a speaker")
        tid = t["turn_id"].to_numpy()
        start, end = t["start"].to_numpy(), t["end"].to_numpy()
        k = np.arange(1, n)
        self_prev = np.array([tid[i - 2] if i >= 2 else None for i in k], dtype=object)
        frames.append(pd.DataFrame({
            "pair_id": [f"{r}__after" for r in tid[k]],
            "session_key": t["session_key"].iloc[0], "interaction_key": ik,
            "partner_turn_id": tid[k - 1], "reply_turn_id": tid[k],
            "self_previous_turn_id": self_prev,
            "partner_speaker": spk[k - 1], "replier": spk[k],
            "reply_turn_position": k, "turns_in_interaction": n,
            "gap_s": start[k] - end[k - 1],
            "n2_candidates": _n2_counts(n, min_distance),
        }))
    if not frames:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    return pd.concat(frames, ignore_index=True)[PAIR_COLUMNS]


__all__ = ["PAIR_COLUMNS", "build_pairs", "usable_n2_counts"]
