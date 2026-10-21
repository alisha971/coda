"""Choosing and cutting a few short listening samples for the EDA notebook.

The rule is fixed and seeded, and looks only at channel quality and turn
density, never at any delivery measure: one interaction per recording site,
plus the interaction with the most overlapping speech and the one with the most
clipped runs. Interactions are limited in length so each clip costs a small
download. The clips are for listening and for checking the measurements by ear;
they are not used in any analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def busiest_window(starts, width: float) -> float:
    """Start of the ``width``-second window containing the most turn starts."""
    s = np.sort(np.asarray(starts, dtype=float))
    if s.size == 0:
        return 0.0
    counts = [((s >= a) & (s < a + width)).sum() for a in s]
    return float(np.floor(s[int(np.argmax(counts))]))


def choose_samples(qc: pd.DataFrame, turns: pd.DataFrame, seed: int,
                   max_minutes: float = 3.0, width: float = 20.0) -> pd.DataFrame:
    if "duration" not in turns.columns:
        raise ValueError("choose_samples needs the turns table, including duration")
    ok = qc[qc["failure"].isna()]
    inter = ok.groupby("interaction_key").agg(
        n=("file_id", "size"), dur=("duration_s", "max"), overlap=("overlap_s", "max"),
        flat=("flat_top_runs", "sum"), site=("vendor_id", "first")).reset_index()
    inter = inter[(inter["n"] == 2) & (inter["dur"] <= max_minutes * 60)]
    picks: list[tuple[str, str]] = []
    rng = np.random.default_rng(seed)
    for site, g in inter.groupby("site"):
        picks.append((f"site V{site}", str(rng.choice(sorted(g["interaction_key"])))))
    main_all = turns[~turns["embedded"]]
    if len(inter):
        top = inter.sort_values("overlap").iloc[-1]["interaction_key"]
        picks.append(("most overlapping speech", top))
        if inter["flat"].max() > 0:
            clip = inter.sort_values("flat").iloc[-1]["interaction_key"]
            picks.append(("clipped loud transient", clip))
        # Extended turns carry most of the speech time in this corpus, so one
        # clip is placed on them. Illustrative only: no analysis selects on
        # turn length, and the sample is never chosen this way.
        med = (main_all[main_all["interaction_key"].isin(inter["interaction_key"])]
               .groupby("interaction_key")["duration"].median())
        if len(med):
            picks.append(("longest turns", str(med.idxmax())))
    seen, rows = set(), []
    main = turns[~turns["embedded"]]
    for label, ik in picks:
        if ik in seen:
            continue
        seen.add(ik)
        t = main[main["interaction_key"] == ik]
        if label == "longest turns" and len(t):
            start = float(np.floor(t.loc[t["duration"].idxmax(), "start"]))
        else:
            start = busiest_window(t["start"], width)
        rows.append({"label": label, "interaction_key": ik, "start_s": start, "length_s": width})
    return pd.DataFrame(rows)


def cut_stereo(a: np.ndarray, b: np.ndarray, sr: int, start_s: float,
               length_s: float) -> np.ndarray:
    """Two mono channels to one stereo clip: first speaker left, second right."""
    i, j = int(start_s * sr), int((start_s + length_s) * sr)
    n = min(len(a), len(b))
    i, j = min(i, n), min(j, n)
    return np.stack([a[i:j], b[i:j]], axis=1)


__all__ = ["busiest_window", "choose_samples", "cut_stereo"]
