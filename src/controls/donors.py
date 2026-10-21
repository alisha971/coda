"""The two negative controls: N1 and N2 donor assignments.

A donor is a turn whose delivery vector replaces the real partner turn's vector
(A) while everything else about the pair stays: the reply, Z (including the
real partner's text), the split, the model and its λ search. Refitting under
donors gives the null distribution Δ is referenced to — the matched-donor
form of a conditional randomisation test (Candès et al. 2018; Berrett et al.
2020): it is valid to the extent that a donor is exchangeable with the real
partner vector given Z, which the β = 0 calibration (G1) checks empirically.

**N1 — cross-dyad matched.** A partner turn from a **different session in the
same split**, matched on partner-turn duration decile (deciles computed within
the split) and partner word count (exact match, otherwise the nearest
word count inside the same decile). Destroys the pairing; preserves
dimensionality, scale, marginals and length structure.

**N2 — within-dyad distant.** A turn by the **same partner speaker in the same
interaction, at least 10 turns away** (turn distance is defined
only inside an interaction). Destroys local adjacency only.

Draws are seeded and saved (turn ids per pair and draw), so every null is
exactly reproducible. :func:`assert_donor_validity` checks every registered
property and is run on every assignment before it is used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


class DonorError(AssertionError):
    """A donor assignment violates a registered property of its scheme."""


def duration_deciles(values: pd.Series) -> pd.Series:
    """Decile 0-9 of each value within its own set (ties broken by rank)."""
    r = values.rank(method="first")
    return np.floor((r - 1) / len(values) * 10).clip(0, 9).astype(int)


def draw_n1(analysis: pd.DataFrame, turns: pd.DataFrame, n_draws: int, seed: int,
            ) -> tuple[np.ndarray, pd.DataFrame]:
    """N1 donor turn ids (n_pairs × n_draws) and a per-pair matching report."""
    t = turns.set_index("turn_id")
    a = analysis.reset_index(drop=True)
    a = a.assign(p_dur=t.loc[a["partner_turn_id"], "duration"].to_numpy(),
                 p_words=t.loc[a["partner_turn_id"], "n_words"].to_numpy())
    rng = np.random.default_rng(seed)
    out = np.empty((len(a), n_draws), dtype=object)
    report = []
    for split, idx in a.groupby("split").groups.items():
        sub = a.loc[idx]
        pool = sub.drop_duplicates("partner_turn_id")[["partner_turn_id", "session_key",
                                                       "p_dur", "p_words"]].copy()
        pool["decile"] = duration_deciles(pool["p_dur"]).to_numpy()
        dec_of = dict(zip(pool["partner_turn_id"], pool["decile"], strict=True))
        cells = {k: g for k, g in pool.groupby(["decile", "p_words"])}
        by_decile = {k: g for k, g in pool.groupby("decile")}
        for i, row in sub.iterrows():
            dec = dec_of[row["partner_turn_id"]]
            cell = cells.get((dec, row["p_words"]))
            exact = True
            cand = None if cell is None else cell[cell["session_key"] != row["session_key"]]
            if cand is None or cand.empty:
                exact = False
                d = by_decile[dec]
                d = d[d["session_key"] != row["session_key"]]
                if d.empty:
                    raise DonorError(f"no N1 donor for {row['pair_id']} in split {split}")
                gap = (d["p_words"] - row["p_words"]).abs()
                cand = d[gap == gap.min()]
            ids = cand["partner_turn_id"].to_numpy()
            out[i] = ids[rng.integers(0, len(ids), n_draws)]
            report.append({"pair_id": row["pair_id"], "split": split, "decile": int(dec),
                           "exact_word_match": exact, "candidates": int(len(ids))})
    return out, pd.DataFrame(report)


def draw_n2(analysis: pd.DataFrame, turns: pd.DataFrame, usable_turns: set, n_draws: int,
            seed: int, min_distance: int = 10) -> np.ndarray:
    """N2 donor turn ids (n_pairs × n_draws)."""
    main = turns[~turns["embedded"].astype(bool)]
    by_spk = {k: g for k, g in main.groupby(["interaction_key", "speaker"])}
    t = main.set_index("turn_id")
    rng = np.random.default_rng(seed)
    a = analysis.reset_index(drop=True)
    out = np.empty((len(a), n_draws), dtype=object)
    for i, row in a.iterrows():
        g = by_spk[(row["interaction_key"], row["partner_speaker"])]
        k = int(t.at[row["partner_turn_id"], "turn_index"])
        ok = ((g["turn_index"].astype(int) - k).abs() >= min_distance) & \
            g["turn_id"].isin(usable_turns)
        ids = g.loc[ok, "turn_id"].to_numpy()
        if len(ids) == 0:
            raise DonorError(f"no usable N2 donor for {row['pair_id']}")
        out[i] = ids[rng.integers(0, len(ids), n_draws)]
    return out


def assert_donor_validity(analysis: pd.DataFrame, donors: np.ndarray, scheme: str,
                          turns: pd.DataFrame, min_distance: int = 10) -> None:
    """Every registered property of the scheme, checked for every pair and draw."""
    a = analysis.reset_index(drop=True)
    if donors.shape[0] != len(a):
        raise DonorError("donor matrix does not match the pairs")
    t = turns.set_index("turn_id")
    flat = donors.ravel()
    partner = np.repeat(a["partner_turn_id"].to_numpy(), donors.shape[1])
    reply = np.repeat(a["reply_turn_id"].to_numpy(), donors.shape[1])
    if (flat == partner).any():
        raise DonorError("a donor is the true partner turn")
    if (flat == reply).any():
        raise DonorError("a donor is the pair's own reply")
    d_sess = t.loc[flat, "session_key"].to_numpy()
    p_sess = np.repeat(a["session_key"].to_numpy(), donors.shape[1])
    split_of = a.drop_duplicates("session_key").set_index("session_key")["split"]
    d_split = pd.Series(d_sess).map(split_of).to_numpy()
    if (d_split != np.repeat(a["split"].to_numpy(), donors.shape[1])).any():
        raise DonorError("a donor comes from a different split")
    if scheme == "N1_cross_dyad":
        if (d_sess == p_sess).any():
            raise DonorError("an N1 donor comes from the pair's own session")
    elif scheme == "N2_within_dyad_distant":
        if (d_sess != p_sess).any():
            raise DonorError("an N2 donor comes from another session")
        if (t.loc[flat, "interaction_key"].to_numpy()
                != np.repeat(a["interaction_key"].to_numpy(), donors.shape[1])).any():
            raise DonorError("an N2 donor comes from another interaction")
        if (t.loc[flat, "speaker"].to_numpy()
                != np.repeat(a["partner_speaker"].to_numpy(), donors.shape[1])).any():
            raise DonorError("an N2 donor is not by the partner speaker")
        dist = np.abs(t.loc[flat, "turn_index"].astype(int).to_numpy()
                      - t.loc[partner, "turn_index"].astype(int).to_numpy())
        if (dist < min_distance).any():
            raise DonorError(f"an N2 donor is fewer than {min_distance} turns away")
    else:
        raise ValueError(f"unknown donor scheme {scheme!r}")


def save_donors(path: Path, donors: np.ndarray, pair_ids, scheme: str, seed: int) -> str:
    """Write a donor matrix with its provenance; returns its content hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps({"pairs": list(map(str, pair_ids)),
                       "donors": donors.astype(str).tolist()}).encode()
    h = hashlib.sha256(blob).hexdigest()[:12]
    np.savez_compressed(path, donors=donors.astype(str), pair_ids=np.asarray(pair_ids, str),
                        scheme=scheme, seed=seed, content_hash=h)
    return h


def load_donors(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {"donors": z["donors"].astype(object), "pair_ids": z["pair_ids"],
            "scheme": str(z["scheme"]), "seed": int(z["seed"]),
            "content_hash": str(z["content_hash"])}


__all__ = ["DonorError", "duration_deciles", "draw_n1", "draw_n2", "assert_donor_validity",
           "save_donors", "load_donors"]
