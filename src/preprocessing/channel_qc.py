"""Channel quality control and the G0 verdict.

A **channel** is one participant's audio file for one interaction. QC has two
layers, kept separate on purpose:

1. **Objective failures.** A channel fails if it is missing or unreadable, is
   not 48 kHz mono, differs in duration from its partner file by more than
   ``max_pair_duration_mismatch_s``, or has no VAD speech or no timed words.
   These are defects, not judgements, and they decide exclusion.
2. **Continuous measures.** Reported as distributions in the audit notebook.
   Any cut on them is stated with its reason, because a threshold chosen
   after looking would be a tuning path into the sample.

The continuous measures address what is specific to this corpus. The released
audio is peak-normalised per file and echo-suppressed to remove the partner's
voice. Echo suppression can also attenuate a speaker while both talk at once,
which would corrupt intensity and create silences that are not pauses. Three
measures target this directly, each computed on 10 ms frames and split by
who is speaking according to the two speakers' VAD:

- ``snr_db``: own speech level minus the level when neither speaks;
- ``bleed_db``: own-channel level while only the partner speaks, relative to
  own speech (strongly negative means little of the partner leaks in);
- ``overlap_drop_db``: own-channel level while both speak, relative to own
  speech alone (near 0 means overlap is not suppressed; strongly negative
  means the speaker's own voice is being gated during double talk).

Terms: **VAD** (voice activity detection) marks when a speaker is talking.
**SNR** (signal-to-noise ratio) compares speech level with background level.
**dBFS** is level relative to digital full scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.tracks import Tracks

QC_OBJECTIVE_FAILURES = (
    "missing", "unreadable", "wrong_sample_rate", "not_mono",
    "duration_mismatch", "no_vad_speech", "no_timed_words",
)


def vad_mask(vad: pd.DataFrame, n_frames: int, step_s: float) -> np.ndarray:
    """Boolean per-frame speech mask from VAD segments."""
    m = np.zeros(n_frames, dtype=bool)
    for s, e in vad[["start", "end"]].itertuples(index=False):
        a, b = int(np.floor(s / step_s)), int(np.ceil(e / step_s))
        m[max(a, 0):min(b, n_frames)] = True
    return m


def _median(x: np.ndarray) -> float:
    return float(np.median(x)) if x.size else float("nan")


def flat_top_runs(x: np.ndarray, min_run: int = 3) -> int:
    """Runs of at least ``min_run`` consecutive samples pinned at the file's peak.

    Peak normalisation hides clipping from a simple threshold, because the
    peak is rescaled; clipping that happened before normalisation survives as
    flat runs at the new peak.
    """
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak == 0:
        return 0
    at_peak = np.abs(x) >= peak * (1 - 1e-6)
    edges = np.diff(np.concatenate(([0], at_peak.astype(np.int8), [0])))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return int((lengths >= min_run).sum())


def channel_measures(x: np.ndarray, sr: int, tracks: Tracks, own_vad: pd.DataFrame,
                     partner_vad: pd.DataFrame, words: pd.DataFrame) -> dict:
    """Continuous QC measures for one channel."""
    step = tracks.step_s
    lv = tracks.frame_dbfs
    n = lv.size
    own = vad_mask(own_vad, n, step)
    oth = vad_mask(partner_vad, n, step)
    own_only, oth_only, both, none = own & ~oth, oth & ~own, own & oth, ~own & ~oth
    own_level = _median(lv[own_only])

    # F0 tracking inside own-only speech: frames of the pitch track that fall
    # in speech where the partner is silent.
    f0_idx = np.clip(np.round((np.arange(n) * step - tracks.f0_t0) / step).astype(int),
                     0, max(tracks.f0_hz.size - 1, 0))
    f0_at = tracks.f0_hz[f0_idx] if tracks.f0_hz.size else np.zeros(n, np.float32)
    voiced = f0_at > 0
    str_at = tracks.pitch_strength[f0_idx] if tracks.pitch_strength.size else f0_at

    word_time = float((words["end"] - words["start"]).clip(lower=0).sum()) if len(words) else 0.0
    return {
        "duration_s": len(x) / sr,
        "peak_abs": float(np.max(np.abs(x))) if x.size else 0.0,
        "flat_top_runs": flat_top_runs(x),
        "noise_floor_dbfs_p5": float(np.percentile(lv, 5)) if n else float("nan"),
        "level_dbfs_p95": float(np.percentile(lv, 95)) if n else float("nan"),
        "vad_speech_s": float(own.sum() * step),
        "overlap_s": float(both.sum() * step),
        "snr_db": own_level - _median(lv[none]),
        "bleed_db": _median(lv[oth_only]) - own_level,
        "overlap_drop_db": _median(lv[both]) - own_level,
        "voiced_frac_own_speech": (float(voiced[own_only].mean()) if own_only.any()
                                   else float("nan")),
        "median_f0_hz": _median(f0_at[own_only & voiced]),
        "median_pitch_strength": _median(str_at[own_only & voiced]),
        "n_words_timed": int(len(words)),
        "n_words_untimed": int(words.attrs.get("n_words_untimed", 0)),
        "word_time_s": word_time,
    }


def objective_failure(*, exists: bool, readable: bool, sample_rate: int | None,
                      channels: int | None, duration_s: float | None,
                      partner_duration_s: float | None, vad_speech_s: float,
                      n_words_timed: int, expected_sr: int, max_mismatch_s: float) -> str | None:
    """The first objective defect found, or None if the channel is usable."""
    if not exists:
        return "missing"
    if not readable:
        return "unreadable"
    if sample_rate != expected_sr:
        return "wrong_sample_rate"
    if channels != 1:
        return "not_mono"
    if (duration_s is None or partner_duration_s is None
            or abs(duration_s - partner_duration_s) > max_mismatch_s):
        return "duration_mismatch"
    if vad_speech_s <= 0:
        return "no_vad_speech"
    if n_words_timed <= 0:
        return "no_timed_words"
    return None


def usable_interactions(channel_qc: pd.DataFrame) -> pd.Series:
    """True for an interaction whose two channels both passed."""
    ok = channel_qc.assign(ok=channel_qc["failure"].isna())
    g = ok.groupby("interaction_key")["ok"]
    return (g.all() & (g.size() == 2)).rename("usable")


def usable_sessions(channel_qc: pd.DataFrame) -> pd.Series:
    """True for a session with at least one usable interaction."""
    inter = usable_interactions(channel_qc).reset_index()
    inter["session_key"] = inter["interaction_key"].str.rsplit("_I", n=1).str[0]
    return inter.groupby("session_key")["usable"].any()


def relative_log2_f0(paths, channel_qc: pd.DataFrame,
                     bins: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Pitch relative to each channel's own median, in octaves (log2 ratio).

    Pitch trackers make octave errors: a frame reported at half or double the
    true F0. They show up as mass near -1 and +1 in this distribution, far from
    the speaker's typical range. The log-F0 range target (p95 - p5) is
    sensitive to them, so their prevalence is reported before any target is
    computed. Descriptive only: nothing is removed here.

    Returns the pooled histogram counts, the bin edges, and per-channel shares
    of voiced frames beyond +/-0.75 octave (a descriptive summary, not a cut).
    """
    from src.features.tracks import Tracks

    bins = np.linspace(-2, 2, 81) if bins is None else bins
    counts = np.zeros(len(bins) - 1)
    rows = []
    ok = channel_qc[channel_qc["failure"].isna()]
    for sk, g in ok.groupby("session_key"):
        path = paths.audio_pass / sk / "tracks.npz"
        if not path.exists():
            continue
        npz = np.load(path)
        for fid in g["file_id"]:
            f0 = Tracks.from_npz(npz, fid).f0_hz
            v = f0[f0 > 0].astype(np.float64)
            if v.size < 10:
                continue
            r = np.log2(v / np.median(v))
            counts += np.histogram(np.clip(r, bins[0], bins[-1]), bins)[0]
            rows.append({"file_id": fid, "session_key": sk, "voiced_frames": int(v.size),
                         "share_below_minus_0_75": float((r < -0.75).mean()),
                         "share_above_plus_0_75": float((r > 0.75).mean())})
    return counts, bins, pd.DataFrame(rows)


def g0_verdict(channel_qc: pd.DataFrame, feature_droppage: float,
               min_pass_rate: float, max_droppage: float) -> dict:
    """G0: at least ``min_pass_rate`` of channels usable and droppage below the maximum.

    Both thresholds are fixed by the specification, not chosen here.
    """
    pass_rate = float((channel_qc["failure"].isna()).mean())
    ok = pass_rate >= min_pass_rate and feature_droppage < max_droppage
    return {
        "channels": int(len(channel_qc)),
        "channel_pass_rate": pass_rate,
        "feature_droppage": float(feature_droppage),
        "min_pass_rate": min_pass_rate,
        "max_droppage": max_droppage,
        "passed": bool(ok),
        "on_failure": "switch the primary corpus to AMI two-party stretches",
    }


__all__ = ["QC_OBJECTIVE_FAILURES", "vad_mask", "flat_top_runs", "channel_measures",
           "objective_failure", "usable_interactions", "usable_sessions",
           "relative_log2_f0", "g0_verdict"]
