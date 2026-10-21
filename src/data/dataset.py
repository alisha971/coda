"""The per-pair information firewall and the common analysis set.

**Firewall.** Every predictor value for a pair is obtained through
:class:`PairFirewall`. A request for the pair's **own reply** audio, SSL
embedding, true duration or delivery values raises :class:`LeakageError`. The
rule is per pair, not global: the same turn is served normally when it is
requested as the *partner*, *self-previous* turn or *donor* of a different pair.
The reply's words (text, word count, syllable count) and its position are
permitted, because the design conditions on what the reply says.

**Temporal order.** Within a pair, the self-previous turn starts before the
partner turn, which starts before the reply; all three lie in one interaction.
:func:`check_temporal_order` asserts this for every pair.

**Analysis set.** One common set of pairs is used for every J1 condition and
every donor scheme, so that droppage cannot differ between conditions:
the reply, partner and self-previous turns must all have their eight dimensions,
the reply and partner text must be non-empty, the pair must have at least one
N2 candidate, and its session must have a partition. Each excluded pair keeps
the first reason that applies, so the exclusion table adds up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TARGET_DIMENSIONS


class LeakageError(RuntimeError):
    """A pair's own reply information was requested as an input."""


#: Information about the reply that may never be an input for its own pair.
REPLY_FORBIDDEN = frozenset({"audio", "ssl", "duration", "start", "end", "delivery",
                             *TARGET_DIMENSIONS})
#: Information about the reply the design conditions on (its words and position).
REPLY_PERMITTED = frozenset({"text", "n_words", "n_syllables", "reply_turn_position",
                             "file_id", "speaker", "turn_id"})

ROLES = ("partner", "self_previous", "reply", "donor")

EXCLUSION_ORDER = (
    "session_not_in_split", "simultaneous_onset", "reply_not_usable", "partner_not_usable",
    "no_self_previous", "self_previous_not_usable", "reply_text_empty", "partner_text_empty",
    "no_n2_candidate",
)


class PairFirewall:
    """Serves turn fields for pairs, refusing a pair's own reply information.

    ``turns`` is indexed by turn id and holds every field that may be served
    (timing, text, delivery dimensions, and paths to audio or SSL features).
    """

    def __init__(self, turns: pd.DataFrame, pairs: pd.DataFrame):
        if pairs["pair_id"].duplicated().any():
            raise ValueError("duplicate pair ids")
        self.turns = turns.set_index("turn_id") if "turn_id" in turns.columns else turns
        if self.turns.index.duplicated().any():
            raise ValueError("duplicate turn ids")
        self.pairs = pairs.set_index("pair_id")
        self.log: list[tuple[str, str, str]] = []

    def _turn_for(self, pair_id: str, role: str, donor_turn: str | None) -> str:
        p = self.pairs.loc[pair_id]
        if role == "partner":
            return p["partner_turn_id"]
        if role == "self_previous":
            tid = p["self_previous_turn_id"]
            if tid is None or (isinstance(tid, float) and np.isnan(tid)):
                raise KeyError(f"{pair_id} has no self-previous turn")
            return tid
        if role == "reply":
            return p["reply_turn_id"]
        if role == "donor":
            if donor_turn is None:
                raise ValueError("a donor request needs the donor turn id")
            if donor_turn == p["reply_turn_id"]:
                raise LeakageError(f"{pair_id}: a pair's own reply cannot be its donor")
            return donor_turn
        raise ValueError(f"unknown role {role!r}")

    def request(self, pair_id: str, role: str, field: str, donor_turn: str | None = None):
        """One field of one turn, in the given role relative to ``pair_id``."""
        if role == "reply" and field not in REPLY_PERMITTED:
            raise LeakageError(
                f"{pair_id}: '{field}' of the pair's own reply is not an allowed input")
        tid = self._turn_for(pair_id, role, donor_turn)
        self.log.append((pair_id, role, field))
        return self.turns.at[tid, field]

    def gather(self, pair_ids, role: str, fields, donor_turns=None) -> pd.DataFrame:
        """Vectorised :meth:`request` for many pairs; same rules."""
        fields = list(fields)
        if role == "reply":
            bad = [f for f in fields if f not in REPLY_PERMITTED]
            if bad:
                raise LeakageError(f"reply fields {bad} are not allowed inputs for their own pair")
        p = self.pairs.loc[list(pair_ids)]
        if role == "partner":
            tids = p["partner_turn_id"]
        elif role == "self_previous":
            tids = p["self_previous_turn_id"]
            if tids.isna().any():
                raise KeyError("some pairs have no self-previous turn")
        elif role == "reply":
            tids = p["reply_turn_id"]
        elif role == "donor":
            tids = pd.Series(np.asarray(donor_turns, dtype=object), index=p.index)
            if (tids.to_numpy() == p["reply_turn_id"].to_numpy()).any():
                raise LeakageError("a pair's own reply was offered as its donor")
        else:
            raise ValueError(f"unknown role {role!r}")
        out = self.turns.loc[tids.to_numpy(), fields]
        out.index = p.index
        return out


def simultaneous_onset(pairs: pd.DataFrame, turns: pd.DataFrame) -> pd.Series:
    """Pairs whose context does not strictly precede the reply because two turns start
    at exactly the same instant.

    Turn construction sorts units by start time, so a reply can never start before
    its partner turn, but it can start at the same timestamp: both speakers' first
    words carry the same time. Such a pair has no temporal precedence. It is
    excluded with this reason; it is not a firewall breach. The same holds for a
    self-previous turn starting together with the partner turn.
    """
    t = turns.set_index("turn_id")["start"].astype(float)
    ps = pairs["partner_turn_id"].map(t)
    tie = ps.eq(pairs["reply_turn_id"].map(t))
    has = pairs["self_previous_turn_id"].notna()
    tie |= has & pairs["self_previous_turn_id"].map(t).eq(ps)
    return tie.fillna(False).astype(bool)


def check_temporal_order(pairs: pd.DataFrame, turns: pd.DataFrame) -> None:
    """Raise if any pair's predictors do not precede its reply."""
    t = turns.set_index("turn_id")
    rs = t.loc[pairs["reply_turn_id"], ["start", "interaction_key"]].to_numpy()
    ps = t.loc[pairs["partner_turn_id"], ["start", "interaction_key"]].to_numpy()
    if not (ps[:, 0].astype(float) < rs[:, 0].astype(float)).all():
        raise LeakageError("a partner turn does not start before its reply")
    if not (ps[:, 1] == rs[:, 1]).all():
        raise LeakageError("a partner turn lies in a different interaction from its reply")
    has = pairs["self_previous_turn_id"].notna().to_numpy()
    if has.any():
        ss = t.loc[pairs.loc[has, "self_previous_turn_id"], ["start", "interaction_key"]].to_numpy()
        if not (ss[:, 0].astype(float) < ps[has, 0].astype(float)).all():
            raise LeakageError("a self-previous turn does not start before the partner turn")
        if not (ss[:, 1] == rs[has, 1]).all():
            raise LeakageError("a self-previous turn lies in a different interaction")


def build_analysis_set(pairs: pd.DataFrame, turns: pd.DataFrame, prosody: pd.DataFrame,
                       split_manifest: pd.DataFrame) -> pd.DataFrame:
    """Every candidate pair with its partition, inclusion flag and exclusion reason."""
    from src.config import DataConfig
    from src.data.pairs import usable_n2_counts

    ties = simultaneous_onset(pairs, turns)
    check_temporal_order(pairs[~ties], turns)     # strict for every other pair
    usable = prosody.set_index("turn_id")["usable"].astype(bool)
    n2_ok = usable_n2_counts(pairs, turns, set(usable.index[usable.to_numpy()]),
                             DataConfig().n2_min_turn_distance)
    text = turns.set_index("turn_id")["text"].fillna("").astype(str).str.strip()
    part = split_manifest.set_index("session_key")["split"]
    out = pairs.copy()
    out["split"] = out["session_key"].map(part)
    checks = {
        "session_not_in_split": out["split"].isna(),
        "simultaneous_onset": ties,
        "reply_not_usable": ~out["reply_turn_id"].map(usable).fillna(False).astype(bool),
        "partner_not_usable": ~out["partner_turn_id"].map(usable).fillna(False).astype(bool),
        "no_self_previous": out["self_previous_turn_id"].isna(),
        "self_previous_not_usable": ~out["self_previous_turn_id"].map(usable)
        .fillna(False).astype(bool),
        "reply_text_empty": out["reply_turn_id"].map(text).fillna("").eq(""),
        "partner_text_empty": out["partner_turn_id"].map(text).fillna("").eq(""),
        # a usable N2 donor, by the donor draw's own rule (not the structural count)
        "no_n2_candidate": n2_ok.to_numpy() <= 0,
    }
    reason = pd.Series(None, index=out.index, dtype=object)
    for name in EXCLUSION_ORDER:
        reason = reason.where(reason.notna() | ~checks[name], name)
    out["exclusion"] = reason
    out["included"] = reason.isna()
    return out


def exclusion_table(analysis: pd.DataFrame) -> pd.DataFrame:
    """Counts by exclusion reason and partition; the rows add up to all candidates."""
    reason = analysis["exclusion"].fillna("included")
    tab = pd.crosstab(reason, analysis["split"].fillna("no partition"), margins=True)
    order = ["included", *[r for r in EXCLUSION_ORDER if r in tab.index], "All"]
    return tab.reindex([r for r in order if r in tab.index])


__all__ = ["LeakageError", "PairFirewall", "REPLY_FORBIDDEN", "REPLY_PERMITTED",
           "EXCLUSION_ORDER", "simultaneous_onset", "check_temporal_order", "build_analysis_set",
           "exclusion_table"]
