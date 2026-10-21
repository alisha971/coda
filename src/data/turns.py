"""Turns from word timestamps (PREREGISTRATION).

The corpus ships timed words per participant, not turns, so turns are built
here in three steps:

1. **Inter-pausal units (IPUs).** One speaker's consecutive words are joined
   while the silence between them is shorter than 50 ms. An IPU is a stretch
   of speech bounded by silence, the unit the cited feature set was defined on.
2. **Turns.** A turn is a maximal run of one speaker's IPUs with no speech
   from the partner *between* them.
3. **Embedded units.** A partner IPU lying entirely inside the current turn's
   extent (an overlapped "yeah" or "mm-hm") does not take the floor. It is kept
   as an ``embedded`` turn so it is visible and countable, but it neither ends
   the host turn nor forms a pair.

Both participants' files of one interaction are cut from one recording over the
same interval, so their timestamps share a zero. Channel QC checks this
(``max_pair_duration_mismatch_s``).
"""

from __future__ import annotations

import pandas as pd

#: Silence that separates inter-pausal units, in seconds.
IPU_GAP_S = 0.050

TURN_COLUMNS = [
    "turn_id", "session_key", "interaction_key", "speaker", "turn_index",
    "embedded", "host_turn_id", "start", "end", "duration", "n_words",
    "n_ipus", "text",
]


def build_ipus(words: pd.DataFrame, gap_s: float = IPU_GAP_S) -> pd.DataFrame:
    """Group one speaker's timed words into inter-pausal units."""
    if words.empty:
        return pd.DataFrame(columns=["start", "end", "n_words", "text"])
    w = words.sort_values("start", kind="stable").reset_index(drop=True)
    # A new unit starts where the silence since the running end reaches the gap.
    running_end = w["end"].cummax().shift(fill_value=-float("inf"))
    new_unit = (w["start"] - running_end) >= gap_s
    uid = new_unit.cumsum()
    g = w.groupby(uid, sort=True)
    return pd.DataFrame({
        "start": g["start"].min().to_numpy(),
        "end": g["end"].max().to_numpy(),
        "n_words": g.size().to_numpy(),
        "text": g["word"].agg(lambda s: " ".join(x.strip() for x in s)).to_numpy(),
    })


def build_turns(words_by_speaker: dict[str, pd.DataFrame], interaction_key: str,
                session_key: str, gap_s: float = IPU_GAP_S) -> pd.DataFrame:
    """Turns for one interaction from each speaker's timed words."""
    if len(words_by_speaker) != 2:
        raise ValueError(f"{interaction_key}: expected two speakers, got {len(words_by_speaker)}")
    units = []
    for spk, words in words_by_speaker.items():
        ipu = build_ipus(words, gap_s)
        ipu["speaker"] = spk
        units.append(ipu)
    allu = pd.concat(units, ignore_index=True)
    allu = allu.sort_values(["start", "end", "speaker"], kind="stable").reset_index(drop=True)

    turns: list[dict] = []
    embedded: list[dict] = []
    cur: dict | None = None

    def open_turn(u) -> dict:
        return {"speaker": u.speaker, "start": u.start, "end": u.end,
                "n_words": int(u.n_words), "n_ipus": 1, "texts": [u.text]}

    for u in allu.itertuples(index=False):
        if cur is None:
            cur = open_turn(u)
        elif u.speaker == cur["speaker"]:
            cur["end"] = max(cur["end"], u.end)
            cur["n_words"] += int(u.n_words)
            cur["n_ipus"] += 1
            cur["texts"].append(u.text)
        elif u.end <= cur["end"]:
            embedded.append({"speaker": u.speaker, "start": u.start, "end": u.end,
                             "n_words": int(u.n_words), "n_ipus": 1, "texts": [u.text],
                             "host": len(turns)})
        else:
            turns.append(cur)
            cur = open_turn(u)
    if cur is not None:
        turns.append(cur)

    rows = []
    for i, t in enumerate(turns):
        rows.append({
            "turn_id": f"{interaction_key}_T{i:04d}", "turn_index": i,
            "embedded": False, "host_turn_id": None, **_common(t),
        })
    for j, t in enumerate(embedded):
        rows.append({
            "turn_id": f"{interaction_key}_E{j:04d}", "turn_index": None,
            "embedded": True, "host_turn_id": f"{interaction_key}_T{t['host']:04d}",
            **_common(t),
        })
    out = pd.DataFrame(rows, columns=[c for c in TURN_COLUMNS
                                      if c not in ("session_key", "interaction_key")])
    out.insert(1, "session_key", session_key)
    out.insert(2, "interaction_key", interaction_key)
    out["turn_index"] = out["turn_index"].astype("Int64")
    # Explicit dtypes: an interaction with no words gives an empty frame, and a
    # concatenation including it would otherwise turn these columns into objects.
    out = out.astype({"embedded": bool, "start": float, "end": float, "duration": float,
                      "n_words": int, "n_ipus": int})
    return out[TURN_COLUMNS]


def _common(t: dict) -> dict:
    return {
        "speaker": t["speaker"], "start": float(t["start"]), "end": float(t["end"]),
        "duration": float(t["end"] - t["start"]), "n_words": t["n_words"],
        "n_ipus": t["n_ipus"], "text": " ".join(t["texts"]),
    }


__all__ = ["IPU_GAP_S", "TURN_COLUMNS", "build_ipus", "build_turns"]
