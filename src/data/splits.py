"""Session-level train/validation/test splits.

The unit of splitting is the session (one dyad). The primary split
is **speaker-disjoint**: no participant appears in more than one partition.
Sessions are first joined into groups that must not be separated:

- sessions sharing a participant (the speaker graph). Under the
  one-session-per-speaker selection this should never link two
  sessions; it is checked directly rather than assumed, because the corpus
  documents rare participant-identifier errors;
- near-duplicate sessions, detected from transcript word shingles.

Groups are placed in a seeded random order and assigned greedily to the
partition furthest below its target share, so every group lands whole in one
partition. The result is checked by :func:`assert_disjoint`, which raises on any
shared participant, shared session or split near-duplicate group.

**The dyad-disjoint contrast** groups by session only. On a sample where every
speaker belongs to one session it is speaker-disjoint as well (the contrast is
degenerate there); :func:`split_report` states how many speakers it shares across partitions
so the degeneracy is visible in every run.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

PARTITIONS = ("train", "val", "test")


class SplitLeakage(AssertionError):
    """A participant, session or near-duplicate group crosses partitions."""


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def shingles(text: str, k: int) -> set[str]:
    words = text.lower().split()
    return {" ".join(words[i:i + k]) for i in range(max(len(words) - k + 1, 0))}


def near_duplicate_links(session_text: dict[str, str], k: int,
                         threshold: float) -> list[tuple[str, str, float]]:
    """Session pairs whose transcript shingle sets have Jaccard >= ``threshold``.

    Quadratic in sessions; at a few hundred sessions this takes seconds.
    """
    keys = sorted(session_text)
    sets = {s: shingles(session_text[s], k) for s in keys}
    links = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            sa, sb = sets[a], sets[b]
            if not sa or not sb:
                continue
            j = len(sa & sb) / len(sa | sb)
            if j >= threshold:
                links.append((a, b, j))
    return links


def session_groups(sessions: pd.DataFrame, near_duplicates: list[tuple[str, str, float]] = (),
                   by_speaker: bool = True) -> pd.Series:
    """Group label per session: sessions sharing a speaker or a near-duplicate link.

    ``sessions`` has one row per (session_key, participant_key).
    """
    uf = _UnionFind()
    for sk in sessions["session_key"].unique():
        uf.find(sk)
    if by_speaker:
        for _, g in sessions.groupby("participant_key"):
            ss = sorted(g["session_key"].unique())
            for other in ss[1:]:
                uf.union(ss[0], other)
    for a, b, _ in near_duplicates:
        uf.union(a, b)
    keys = sorted(sessions["session_key"].unique())
    return pd.Series({k: uf.find(k) for k in keys}, name="group")


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------


def assign_groups(groups: pd.Series, fractions: dict[str, float], seed: int) -> pd.Series:
    """Partition per session. Whole groups only; seeded order; greedy fill."""
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1, got {fractions}")
    sizes = groups.value_counts().sort_index()
    if len(sizes) < len(fractions):
        raise ValueError(f"{len(sizes)} groups cannot fill {len(fractions)} partitions")
    order = np.random.default_rng(seed).permutation(sizes.index.to_numpy())
    total = int(sizes.sum())
    target = {p: fractions[p] * total for p in fractions}
    filled = {p: 0 for p in fractions}
    group_to_part: dict[str, str] = {}
    for g in order:
        # The partition with the largest shortfall relative to its target.
        part = max(fractions, key=lambda p: (target[p] - filled[p]) / max(target[p], 1e-9))
        group_to_part[g] = part
        filled[part] += int(sizes[g])
    return groups.map(group_to_part).rename("split")


def make_split(sessions: pd.DataFrame, fractions: dict[str, float], seed: int,
               scheme: str = "speaker_disjoint",
               near_duplicates: list[tuple[str, str, float]] = ()) -> pd.DataFrame:
    """The split manifest: one row per session with its group and partition."""
    if scheme == "speaker_disjoint":
        groups = session_groups(sessions, near_duplicates, by_speaker=True)
    elif scheme == "dyad_disjoint":
        groups = session_groups(sessions, near_duplicates, by_speaker=False)
    else:
        raise ValueError(f"unknown split scheme {scheme!r}")
    split = assign_groups(groups, fractions, seed)
    out = pd.DataFrame({"session_key": groups.index, "group": groups.to_numpy(),
                        "split": split.to_numpy()})
    out["scheme"] = scheme
    out["seed"] = seed
    if scheme == "speaker_disjoint":
        assert_disjoint(out, sessions, near_duplicates)
    else:
        assert_disjoint(out, sessions, near_duplicates, check_speakers=False)
    return out


def assert_disjoint(manifest: pd.DataFrame, sessions: pd.DataFrame,
                    near_duplicates: list[tuple[str, str, float]] = (),
                    check_speakers: bool = True) -> None:
    """Raise :class:`SplitLeakage` if anything crosses partitions."""
    if manifest["session_key"].duplicated().any():
        raise SplitLeakage("a session appears more than once in the split manifest")
    if not set(manifest["split"]) <= set(PARTITIONS):
        raise SplitLeakage(f"unknown partitions {set(manifest['split']) - set(PARTITIONS)}")
    part = manifest.set_index("session_key")["split"]
    missing = set(sessions["session_key"]) - set(part.index)
    if missing:
        raise SplitLeakage(f"{len(missing)} sessions have no partition")
    if check_speakers:
        sp = sessions.assign(split=sessions["session_key"].map(part))
        n_parts = sp.groupby("participant_key")["split"].nunique()
        bad = n_parts[n_parts > 1]
        if len(bad):
            raise SplitLeakage(f"{len(bad)} participants appear in more than one partition, "
                               f"e.g. {list(bad.index[:3])}")
    for a, b, _ in near_duplicates:
        if part.get(a) != part.get(b):
            raise SplitLeakage(f"near-duplicate sessions {a} and {b} are in different partitions")


def split_report(manifest: pd.DataFrame, sessions: pd.DataFrame,
                 pairs_per_session: pd.Series | None = None) -> dict:
    """Counts per partition and the cross-partition speaker check."""
    part = manifest.set_index("session_key")["split"]
    sp = sessions.assign(split=sessions["session_key"].map(part))
    shared = int((sp.groupby("participant_key")["split"].nunique() > 1).sum())
    rep = {"scheme": manifest["scheme"].iloc[0], "seed": int(manifest["seed"].iloc[0]),
           "sessions": manifest.groupby("split").size().reindex(PARTITIONS, fill_value=0)
           .astype(int).to_dict(),
           "speakers": sp.groupby("split")["participant_key"].nunique()
           .reindex(PARTITIONS, fill_value=0).astype(int).to_dict(),
           "groups_with_more_than_one_session": int((manifest["group"].value_counts() > 1).sum()),
           "speakers_shared_across_partitions": shared,
           "manifest_hash": manifest_hash(manifest)}
    if pairs_per_session is not None:
        rep["pairs"] = (manifest.assign(n=manifest["session_key"].map(pairs_per_session)
                                        .fillna(0)).groupby("split")["n"].sum()
                        .reindex(PARTITIONS, fill_value=0).astype(int).to_dict())
    return rep


def manifest_hash(manifest: pd.DataFrame) -> str:
    """Identifier of a split assignment, recorded in every run record."""
    rows = sorted(zip(manifest["session_key"], manifest["split"], strict=True))
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:12]


__all__ = ["PARTITIONS", "SplitLeakage", "shingles", "near_duplicate_links", "session_groups",
           "assign_groups", "make_split", "assert_disjoint", "split_report", "manifest_hash"]
