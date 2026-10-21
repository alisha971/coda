"""Descriptive summaries for the exploratory notebooks.

Exploratory only. Nothing here relates a partner turn's delivery to its reply's
delivery: that relationship is the confirmatory question (H1), and examining it
before calibration would open a path from the data into analysis choices.
Everything here describes speakers, conversations, turn-taking and the
measurement itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.tracks import Tracks
from src.paths import Paths


def talk_share(turns: pd.DataFrame) -> pd.DataFrame:
    """Each speaker's share of speaking time in each interaction.

    Uses all turn units, embedded ones included, because backchannels are part
    of how much a person talks. 0.5 means a balanced conversation.
    """
    t = turns.assign(d=turns["duration"].clip(lower=0))
    per = t.groupby(["session_key", "interaction_key", "speaker"])["d"].sum().rename("seconds")
    per = per.reset_index()
    total = per.groupby("interaction_key")["seconds"].transform("sum")
    per["share"] = np.where(total > 0, per["seconds"] / total, np.nan)
    return per


def turn_rates(turns: pd.DataFrame) -> pd.DataFrame:
    """Words per second over each non-embedded turn's span (pauses included).

    A coarse speaking-rate view for exploration. The target dimension
    *articulation rate* excludes pauses and is computed in the prosody stage.
    """
    t = turns[~turns["embedded"]].copy()
    t["words_per_s"] = np.where(t["duration"] > 0, t["n_words"] / t["duration"], np.nan)
    t["ends_with_question_mark"] = t["text"].str.strip().str.endswith("?")
    return t


def file_prosody_summary(paths: Paths, channel_qc: pd.DataFrame) -> pd.DataFrame:
    """Per usable file: pitch and intensity summaries over voiced frames.

    Voiced frames only: pitch is tracked inside the speaker's own speech, so
    voiced frames are the speaker's own voice. Intensity is read at the same
    times. Returns median F0, log-F0 p5/p95 span, median intensity and counts.
    """
    rows = []
    ok = channel_qc[channel_qc["failure"].isna()]
    for sk, g in ok.groupby("session_key"):
        path = paths.audio_pass / sk / "tracks.npz"
        if not path.exists():
            continue
        npz = np.load(path)
        for r in g.itertuples(index=False):
            tr = Tracks.from_npz(npz, r.file_id)
            voiced = tr.f0_hz > 0
            if voiced.sum() < 10:
                continue
            f0 = tr.f0_hz[voiced].astype(np.float64)
            t = tr.f0_t0 + np.flatnonzero(voiced) * tr.step_s
            k = np.clip(np.round((t - tr.int_t0) / tr.step_s).astype(int), 0,
                        tr.intensity_db.size - 1)
            db = tr.intensity_db[k].astype(np.float64)
            lf = np.log(f0)
            rows.append({
                "file_id": r.file_id, "session_key": sk, "interaction_key": r.interaction_key,
                "participant_key": r.participant_key, "vendor_id": r.vendor_id,
                "voiced_frames": int(voiced.sum()),
                "median_f0_hz": float(np.median(f0)),
                "log_f0_span_p5_p95": float(np.percentile(lf, 95) - np.percentile(lf, 5)),
                "median_intensity_db": float(np.median(db)),
                "intensity_span_p5_p95": float(np.percentile(db, 95) - np.percentile(db, 5)),
            })
    return pd.DataFrame(rows)


def between_group_share(df: pd.DataFrame, value: str, group: str) -> float:
    """Share of the total variance of ``value`` lying between ``group`` means (eta squared).

    Used to show how much of a measure's spread is a property of *who* is
    speaking rather than of the moment. That is the reason the targets are
    standardised per speaker.
    """
    d = df[[value, group]].dropna()
    if d.empty:
        return float("nan")
    grand = d[value].mean()
    total = ((d[value] - grand) ** 2).sum()
    means = d.groupby(group)[value].transform("mean")
    between = ((means - grand) ** 2).sum()
    return float(between / total) if total > 0 else float("nan")


__all__ = ["talk_share", "turn_rates", "file_prosody_summary", "between_group_share"]
