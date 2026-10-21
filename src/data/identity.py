"""Screen for one real person recorded under two participant IDs.

The rule was fixed in the preregistration before the screen ran. Two
candidate-link rules, both within one vendor:

- **L1**: one ID equals the other plus the suffix "A";
- **L2**: both BFI-2 profiles disclosed and all five domain scores equal.

Flagged sessions are grouped in the split exactly like near-duplicate
transcripts; nothing is excluded. The screen's recall is unknown: it is not
identity verification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DOMAINS = ("extraversion_raw", "agreeableness_raw", "conscientiousness_raw",
           "neuroticism_raw", "openness_raw")
UNDISCLOSED = "Undisclosed"


def _profiles(participants: pd.DataFrame) -> pd.DataFrame:
    p = participants.copy()
    p["vendor_id"] = p["vendor_id"].astype(str)
    p["participant_id"] = p["participant_id"].astype(str)
    p["disclosed"] = ~(p[list(DOMAINS)].astype(str) == UNDISCLOSED).any(axis=1)
    p["profile"] = p[list(DOMAINS)].astype(str).agg("|".join, axis=1)
    return p


def analysis_participants(manifest: pd.DataFrame, roles=("pilot", "sample")) -> pd.DataFrame:
    """One row per (session, participant) in the analysis set."""
    m = manifest[manifest["role"].isin(roles)]
    return (m[["role", "session_key", "vendor_id", "participant_id", "participant_key"]]
            .astype(str).drop_duplicates(["session_key", "participant_key"])
            .reset_index(drop=True))


def candidate_links(manifest: pd.DataFrame, participants: pd.DataFrame,
                    roles=("pilot", "sample")) -> pd.DataFrame:
    """Candidate same-person links between IDs in different selected sessions."""
    sel = analysis_participants(manifest, roles).merge(
        _profiles(participants), on=["vendor_id", "participant_id"], how="left")
    sel["disclosed"] = sel["disclosed"].fillna(False).astype(bool)
    rows = []
    for _, g in sel.groupby("vendor_id"):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g.loc[i], g.loc[j]
                if a["session_key"] == b["session_key"]:
                    continue
                ids = sorted([a["participant_id"], b["participant_id"]], key=len)
                if ids[1] == ids[0] + "A":
                    rows.append((a, b, "L1_suffix"))
                if a["disclosed"] and b["disclosed"] and a["profile"] == b["profile"]:
                    rows.append((a, b, "L2_trait_profile"))
    cols = ["rule", "vendor_id", "participant_key_a", "participant_key_b", "session_a",
            "session_b", "role_a", "role_b"]
    return pd.DataFrame([{"rule": r, "vendor_id": a["vendor_id"],
                          "participant_key_a": a["participant_key"],
                          "participant_key_b": b["participant_key"],
                          "session_a": a["session_key"], "session_b": b["session_key"],
                          "role_a": a["role"], "role_b": b["role"]} for a, b, r in rows],
                        columns=cols)


def profile_match_null(manifest: pd.DataFrame, participants: pd.DataFrame,
                       n_perm: int = 1000, seed: int = 20261016,
                       roles=("pilot", "sample")) -> dict:
    """Chance L2 matches: each domain permuted independently within vendor."""
    sel = analysis_participants(manifest, roles).merge(
        _profiles(participants), on=["vendor_id", "participant_id"], how="left")
    sel = sel[sel["disclosed"].fillna(False).astype(bool)].reset_index(drop=True)

    def count(df):
        n = 0
        for _, g in df.groupby("vendor_id"):
            # pairs in different sessions sharing a profile
            for _, h in g.groupby("profile"):
                k = len(h)
                if k > 1:
                    same = h.groupby("session_key").size()
                    n += k * (k - 1) // 2 - int((same * (same - 1) // 2).sum())
        return n

    observed = count(sel)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=int)
    for t in range(n_perm):
        q = sel.copy()
        for d in DOMAINS:
            for _, idx in q.groupby("vendor_id").groups.items():
                q.loc[idx, d] = rng.permutation(q.loc[idx, d].to_numpy())
        q["profile"] = q[list(DOMAINS)].astype(str).agg("|".join, axis=1)
        null[t] = count(q)
    return {"observed": int(observed), "null_mean": float(null.mean()),
            "null_q95": float(np.quantile(null, 0.95)),
            "null_q99": float(np.quantile(null, 0.99)),
            "p_null_ge_observed": float((null >= observed).mean()),
            "disclosed_participants": int(len(sel)), "n_perm": n_perm, "seed": seed}


def as_split_links(links: pd.DataFrame) -> list[tuple[str, str, float]]:
    """Candidate links in the form ``session_groups`` takes (score 1.0)."""
    return sorted({(r.session_a, r.session_b, 1.0) for r in links.itertuples()})


def stage_links(links: list[tuple[str, str, float]],
                stage_sessions) -> list[tuple[str, str, float]]:
    """Candidate links restated for one stage's sessions.

    The screen runs over the whole analysis set, so a link can join a pilot
    session to a sample session. A split is built for one stage only, so such a
    link has no partner in it and must be dropped there. Two sessions of the stage
    joined through sessions outside it remain linked: each connected component is
    restated as links from its first stage member to each other stage member.
    """
    stage = set(stage_sessions)
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in links:
        parent[find(a)] = find(b)
    members: dict[str, list[str]] = {}
    for x in list(parent):
        if x in stage:
            members.setdefault(find(x), []).append(x)
    out = []
    for group in members.values():
        group = sorted(group)
        out += [(group[0], other, 1.0) for other in group[1:]]
    return sorted(out)


__all__ = ["DOMAINS", "analysis_participants", "candidate_links", "profile_match_null",
           "as_split_links", "stage_links"]
