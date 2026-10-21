"""Turn-level prosody from the stored frame tracks (PREREGISTRATION §4).

Two parts:

1. **G0 support** (unchanged since the pilot audit): the voiced-frame count per
   turn and feature droppage, :func:`turn_voicing` and :func:`feature_droppage`.
2. **The eight target dimensions** per non-embedded turn, from the stored
   frame tracks and the turn's timed words, :func:`measure_turn` and
   :func:`compute_turn_prosody`.

The same function measures every turn. Whether a turn later serves as a reply
(target), a partner turn, a self-previous turn or a donor is decided downstream,
behind the per-pair firewall (``src/data/dataset.py``), so nothing about a
turn's role can influence how it is measured.

Operational choices that the specification leaves open are read from
:class:`src.settings.Settings` and recorded with every output:

- **speech frames** are the frames whose centres fall inside the turn's
  inter-pausal units (its words merged at the 50 ms IPU gap); pauses inside the
  turn are measured by the timing block and are excluded from pitch and energy;
- **mean intensity** follows Praat's default "energy" averaging,
  10*log10(mean(10^(dB/10))), unless the settings say otherwise;
- **intensity range** is P95 - P5 over speech frames, by analogy with the
  registered log-F0 range;
- the **boundary slope** is the least-squares slope of ln F0 on time over voiced
  frames in the final 300 ms, and requires its own minimum of voiced frames;
- a **single-IPU turn has an internal pause of exactly 0 s**: a measured value,
  not a missing one.

A turn is usable only when all eight values are finite (missing prosody
drops the turn). Each missing value carries a reason code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import TARGET_DIMENSIONS
from src.features.tracks import Tracks
from src.paths import Paths

#: The eight registered dimensions, in block order (PREREGISTRATION section 4).
TARGET_DIMS: tuple[str, ...] = TARGET_DIMENSIONS


def turn_file_id(interaction_key: str, speaker: str) -> str:
    """The audio file holding a speaker's side of an interaction."""
    return f"{interaction_key}_P{speaker.split('_P', 1)[1]}"


def frames_in(tracks_t0: float, step: float, n: int, start: float, end: float) -> slice:
    """Frame indices whose centres fall inside [start, end)."""
    a = int(np.ceil((start - tracks_t0) / step))
    b = int(np.ceil((end - tracks_t0) / step))
    return slice(max(a, 0), min(max(b, 0), n))


def turn_voicing(turns: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    """Voiced-frame count and voiced fraction for every turn with stored tracks.

    Turns whose session has not been through the audio pass, or whose file
    failed it, get ``has_tracks = False`` rather than a count.
    """
    rows = []
    for sk, t in turns.groupby("session_key", sort=True):
        path = paths.audio_pass / sk / "tracks.npz"
        if not path.exists():
            rows.extend({"turn_id": tid, "has_tracks": False} for tid in t["turn_id"])
            continue
        npz = np.load(path)
        keys = {k.rsplit("__", 1)[0] for k in npz.files}
        cache: dict[str, Tracks] = {}
        for r in t.itertuples(index=False):
            fid = turn_file_id(r.interaction_key, r.speaker)
            if fid not in keys:
                rows.append({"turn_id": r.turn_id, "has_tracks": False})
                continue
            tr = cache.get(fid) or cache.setdefault(fid, Tracks.from_npz(npz, fid))
            sl = frames_in(tr.f0_t0, tr.step_s, tr.f0_hz.size, r.start, r.end)
            f0 = tr.f0_hz[sl]
            rows.append({"turn_id": r.turn_id, "has_tracks": True,
                         "n_frames": int(f0.size), "voiced_frames": int((f0 > 0).sum()),
                         "voiced_frac": float((f0 > 0).mean()) if f0.size else 0.0})
    return pd.DataFrame(rows)


def feature_droppage(pairs: pd.DataFrame, voicing: pd.DataFrame,
                     min_voiced_frames: int) -> dict:
    """Share of candidate-pair turns whose prosody cannot be measured.

    Counted over the distinct turns that appear in a candidate pair as partner
    or reply, among turns with stored tracks. Pairs lost because either turn is
    unmeasurable are reported beside it.
    """
    v = voicing[voicing["has_tracks"]].set_index("turn_id")
    ids = pd.unique(pd.concat([pairs["partner_turn_id"], pairs["reply_turn_id"]]))
    ids = [i for i in ids if i in v.index]
    measurable = v.loc[ids, "voiced_frames"] >= min_voiced_frames
    ok = v["voiced_frames"] >= min_voiced_frames
    in_v = pairs["partner_turn_id"].isin(v.index) & pairs["reply_turn_id"].isin(v.index)
    with_tracks = pairs[in_v]
    both = ok.reindex(with_tracks["partner_turn_id"]).to_numpy() & \
        ok.reindex(with_tracks["reply_turn_id"]).to_numpy()
    return {
        "turns_considered": int(len(ids)),
        "turns_dropped": int((~measurable).sum()),
        "turn_droppage": float((~measurable).mean()) if len(ids) else float("nan"),
        "pairs_considered": int(len(with_tracks)),
        "pairs_lost": int((~both).sum()),
        "pair_loss": float((~both).mean()) if len(with_tracks) else float("nan"),
    }


# --------------------------------------------------------------------------
# The eight target dimensions
# --------------------------------------------------------------------------

#: Reason codes for a missing dimension.
FAIL_NO_TRACKS = "no_tracks"
FAIL_NO_WORDS = "no_words"
FAIL_TOO_FEW_VOICED = "too_few_voiced"
FAIL_BOUNDARY_TOO_FEW_VOICED = "boundary_too_few_voiced"
FAIL_NO_SPEECH_FRAMES = "no_speech_frames"
FAIL_ZERO_PHONATION = "zero_phonation"


class MeasurementNotAvailable(RuntimeError):
    """The configured measurement cannot be computed from the stored tracks."""


#: Overlap handling. ``exclude_overlap_energy`` drops partner-overlapped frames from the two
#: intensity values only; pauses keep the whole turn, because an overlapped gap
#: is still a gap in the replier's own speech.
OVERLAP_OPTIONS = ("all", "exclude_overlap", "exclude_overlap_energy")


@dataclass(frozen=True)
class ProsodySettings:
    """Everything that determines a turn's eight values."""

    min_voiced_frames: int = 10
    boundary_window_s: float = 0.300
    boundary_min_voiced_frames: int = 10
    ipu_gap_s: float = 0.050
    overlap_handling: str = "all"   # all | exclude_overlap | exclude_overlap_energy
    pitch_range: str = "fixed"                # fixed | speaker_adaptive
    syllable_method: str = "rule_based"       # rule_based | dictionary
    intensity_averaging: str = "energy"       # energy | db
    intensity_range: str = "p95_p5"

    @classmethod
    def from_config(cls, features=None, settings=None) -> ProsodySettings:
        from src.config import FeatureConfig
        from src.settings import Settings

        f = features or FeatureConfig()
        s = settings or Settings()
        tm = s.turn_measurement
        return cls(min_voiced_frames=f.min_voiced_frames, boundary_window_s=f.boundary_window_s,
                   boundary_min_voiced_frames=int(tm["boundary_min_voiced_frames"]),
                   overlap_handling=s.overlap_handling, pitch_range=s.pitch_range,
                   syllable_method=s.syllable_method,
                   intensity_averaging=tm["intensity_averaging"],
                   intensity_range=tm["intensity_range"])

    def check(self) -> None:
        if self.overlap_handling not in OVERLAP_OPTIONS:
            raise ValueError(f"unknown overlap handling {self.overlap_handling!r}")
        if self.pitch_range == "speaker_adaptive":
            raise MeasurementNotAvailable(
                "'speaker_adaptive' pitch range changes the Praat floor and ceiling, which "
                "were applied when the tracks were extracted (75-500 Hz). It needs a new "
                "audio pass; it cannot be computed from the stored tracks.")
        if self.pitch_range != "fixed":
            raise ValueError(f"unknown pitch range {self.pitch_range!r}")
        if self.intensity_averaging not in ("energy", "db"):
            raise ValueError(f"unknown intensity averaging {self.intensity_averaging!r}")
        if self.intensity_range != "p95_p5":
            raise ValueError(f"unknown intensity range {self.intensity_range!r}")


def _in_spans(t: np.ndarray, spans: np.ndarray) -> np.ndarray:
    """True where a time lies inside any span [s, e) of a sorted, merged set."""
    if spans.size == 0 or t.size == 0:
        return np.zeros(t.shape, dtype=bool)
    idx = np.searchsorted(spans[:, 0], t, side="right") - 1
    ok = idx >= 0
    out = np.zeros(t.shape, dtype=bool)
    out[ok] = t[ok] < spans[idx[ok], 1]
    return out


def _subtract(intervals: np.ndarray, cover: np.ndarray) -> np.ndarray:
    """Parts of ``intervals`` not covered by ``cover`` (both sorted and merged)."""
    out = []
    for s, e in intervals:
        cur = s
        for cs, ce in cover:
            if ce <= cur or cs >= e:
                continue
            if cs > cur:
                out.append((cur, cs))
            cur = max(cur, ce)
            if cur >= e:
                break
        if cur < e:
            out.append((cur, e))
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def _merge(spans: np.ndarray, gap: float) -> np.ndarray:
    from src.features.tracks import merge_intervals

    return merge_intervals(spans, gap_s=gap)


def measure_turn(start: float, end: float, word_spans: np.ndarray, text: str,
                 f0_t0: float, f0: np.ndarray, int_t0: float, intensity: np.ndarray,
                 step: float, settings: ProsodySettings,
                 partner_spans: np.ndarray | None = None) -> dict:
    """The eight dimensions of one turn, with counts and failure reasons.

    ``word_spans`` are the turn speaker's word intervals inside the turn and
    ``partner_spans`` the partner's word intervals (used only under
    ``exclude_overlap`` and ``exclude_overlap_energy``). All times are on the
    file's axis, in seconds.
    """
    from src.features.syllables import count_syllables

    settings.check()
    out: dict = {d: np.nan for d in TARGET_DIMS}
    fail: dict = {d: None for d in TARGET_DIMS}
    words = np.asarray(word_spans, dtype=np.float64).reshape(-1, 2)
    if words.size == 0:
        return _result(out, {d: FAIL_NO_WORDS for d in TARGET_DIMS}, 0, 0, 0.0, 0)
    ipus = _merge(words, settings.ipu_gap_s)
    partner = (_merge(np.asarray(partner_spans, dtype=np.float64).reshape(-1, 2), 0.0)
               if partner_spans is not None and len(partner_spans) else np.zeros((0, 2)))

    # -- pitch: speech frames only, voiced = F0 > 0 ------------------------
    tf = f0_t0 + np.arange(f0.size) * step
    voiced = _in_spans(tf, ipus) & (f0 > 0)
    n_voiced = int(voiced.sum())
    if n_voiced >= settings.min_voiced_frames:
        lf = np.log(f0[voiced].astype(np.float64))
        out["mean_log_f0"] = float(lf.mean())
        out["log_f0_range_p95_p5"] = float(np.percentile(lf, 95) - np.percentile(lf, 5))
    else:
        fail["mean_log_f0"] = fail["log_f0_range_p95_p5"] = FAIL_TOO_FEW_VOICED
    win = voiced & (tf >= end - settings.boundary_window_s) & (tf < end)
    n_bound = int(win.sum())
    if n_bound >= max(settings.boundary_min_voiced_frames, 2):
        t = tf[win]
        out["log_f0_slope_final_300ms"] = float(
            np.polyfit(t - t.mean(), np.log(f0[win].astype(np.float64)), 1)[0])
    else:
        fail["log_f0_slope_final_300ms"] = FAIL_BOUNDARY_TOO_FEW_VOICED

    # -- energy: speech frames, optionally without the partner's speech ------
    ti = int_t0 + np.arange(intensity.size) * step
    speech_i = _in_spans(ti, ipus)
    if settings.overlap_handling in ("exclude_overlap", "exclude_overlap_energy") \
            and partner.size:
        speech_i &= ~_in_spans(ti, partner)
    db = intensity[speech_i].astype(np.float64)
    db = db[np.isfinite(db)]
    if db.size >= 1:
        if settings.intensity_averaging == "energy":
            out["mean_intensity"] = float(10 * np.log10(np.mean(10 ** (db / 10))))
        else:
            out["mean_intensity"] = float(db.mean())
    else:
        fail["mean_intensity"] = FAIL_NO_SPEECH_FRAMES
    if db.size >= 2:
        out["intensity_range"] = float(np.percentile(db, 95) - np.percentile(db, 5))
    else:
        fail["intensity_range"] = FAIL_NO_SPEECH_FRAMES

    # -- timing: from word timing, not audio level (echo suppression) --
    phonation = float(np.clip(ipus[:, 1] - ipus[:, 0], 0, None).sum())
    n_syll = count_syllables(text, settings.syllable_method)
    if phonation > 0:
        out["articulation_rate"] = n_syll / phonation
    else:
        fail["articulation_rate"] = FAIL_ZERO_PHONATION
    gaps = (np.column_stack([ipus[:-1, 1], ipus[1:, 0]]) if len(ipus) > 1
            else np.zeros((0, 2)))
    duration = float(end - start)
    if settings.overlap_handling == "exclude_overlap" and partner.size and len(gaps):
        kept = _subtract(gaps, partner)
        removed = float((gaps[:, 1] - gaps[:, 0]).sum() - (kept[:, 1] - kept[:, 0]).sum())
        gaps, duration = kept, duration - removed
    pause = np.clip(gaps[:, 1] - gaps[:, 0], 0, None) if len(gaps) else np.zeros(0)
    if duration > 0:
        out["pause_ratio"] = float(pause.sum() / duration)
        out["longest_internal_pause"] = float(pause.max()) if pause.size else 0.0
    else:
        fail["pause_ratio"] = fail["longest_internal_pause"] = FAIL_ZERO_PHONATION
    return _result(out, fail, n_voiced, n_bound, phonation, n_syll)


def _result(values: dict, fail: dict, n_voiced: int, n_bound: int, phonation: float,
            n_syll: int) -> dict:
    row = dict(values)
    row.update({f"fail_{d}": fail[d] for d in TARGET_DIMS})
    row.update(n_voiced=n_voiced, n_boundary_voiced=n_bound, phonation_s=phonation,
               n_syllables=n_syll)
    row["usable"] = bool(all(np.isfinite(values[d]) for d in TARGET_DIMS))
    return row


def compute_turn_prosody(turns: pd.DataFrame, load_tracks, load_words,
                         settings: ProsodySettings) -> pd.DataFrame:
    """The eight dimensions for every non-embedded turn.

    ``load_tracks(session_key, file_id)`` returns :class:`Tracks` or None;
    ``load_words(file_id)`` returns the timed-word table or None. Both are
    injected, so the function runs on synthetic fixtures in tests and on the
    stored audio-pass outputs on Colab.
    """
    settings.check()
    main = turns[~turns["embedded"].astype(bool)]
    rows = []
    for ik, t in main.groupby("interaction_key", sort=True):
        words = {}
        for spk in sorted(turns.loc[turns["interaction_key"] == ik, "speaker"].unique()):
            w = load_words(turn_file_id(ik, spk))
            spans = (w[["start", "end"]].to_numpy(np.float64) if w is not None and len(w)
                     else np.zeros((0, 2)))
            words[spk] = (spans, w)
        for r in t.itertuples(index=False):
            fid = turn_file_id(ik, r.speaker)
            base = {"turn_id": r.turn_id, "session_key": r.session_key,
                    "interaction_key": ik, "speaker": r.speaker, "file_id": fid,
                    "start": float(r.start), "end": float(r.end)}
            tr = load_tracks(r.session_key, fid)
            if tr is None:
                rows.append({**base, **_result({d: np.nan for d in TARGET_DIMS},
                                               {d: FAIL_NO_TRACKS for d in TARGET_DIMS},
                                               0, 0, 0.0, 0)})
                continue
            spans, wdf = words.get(r.speaker, (np.zeros((0, 2)), None))
            inside = (spans[:, 0] >= r.start - 1e-9) & (spans[:, 1] <= r.end + 1e-9)
            text = (" ".join(wdf.loc[inside, "word"].astype(str))
                    if wdf is not None and len(wdf) else "")
            others = [s for s in words if s != r.speaker]
            partner = words[others[0]][0] if others else None
            m = measure_turn(r.start, r.end, spans[inside], text, tr.f0_t0, tr.f0_hz,
                             tr.int_t0, tr.intensity_db, tr.step_s, settings, partner)
            rows.append({**base, **m})
    return pd.DataFrame(rows)


def turn_prosody_from_paths(turns: pd.DataFrame, paths: Paths,
                            settings: ProsodySettings) -> pd.DataFrame:
    """:func:`compute_turn_prosody` reading the audio-pass tracks and raw transcripts."""
    from src.data.seamless import read_words

    cache: dict = {}

    def load_tracks(sk: str, fid: str):
        if sk not in cache:
            cache.clear()   # one session's archive in memory at a time
            path = paths.audio_pass / sk / "tracks.npz"
            cache[sk] = np.load(path) if path.exists() else None
        npz = cache[sk]
        if npz is None or f"{fid}__meta" not in npz.files:
            return None
        return Tracks.from_npz(npz, fid)

    def load_words(fid: str):
        path = paths.raw_transcript / f"{fid}.jsonl"
        return read_words(path) if path.exists() else None

    return compute_turn_prosody(turns.sort_values(["session_key", "interaction_key"]),
                                load_tracks, load_words, settings)


def file_loudness_offset(tracks: Tracks) -> float:
    """Per-file loudness offset: median intensity over voiced frames, in dB.

    The corpus is peak-normalised per file, so absolute level differs between
    files for reasons unrelated to delivery. This offset enters Z so the model
    can absorb that difference.
    """
    voiced = tracks.f0_hz > 0
    if not voiced.any() or tracks.intensity_db.size == 0:
        return float("nan")
    t = tracks.f0_t0 + np.flatnonzero(voiced) * tracks.step_s
    k = np.clip(np.round((t - tracks.int_t0) / tracks.step_s).astype(int), 0,
                tracks.intensity_db.size - 1)
    return float(np.median(tracks.intensity_db[k]))


def droppage_by_dimension(prosody: pd.DataFrame) -> pd.DataFrame:
    """Share of turns missing each dimension, with the most common reason."""
    rows = []
    n = len(prosody)
    for d in TARGET_DIMS:
        miss = ~np.isfinite(prosody[d].astype(float))
        reasons = prosody.loc[miss, f"fail_{d}"].value_counts()
        rows.append({"dimension": d, "turns": n, "missing": int(miss.sum()),
                     "share_missing": float(miss.mean()) if n else float("nan"),
                     "main_reason": reasons.index[0] if len(reasons) else None})
    dropped = ~prosody["usable"].astype(bool)
    rows.append({"dimension": "any (turn dropped)", "turns": n, "missing": int(dropped.sum()),
                 "share_missing": float(dropped.mean()) if n else float("nan"),
                 "main_reason": None})
    return pd.DataFrame(rows)


__all__ = ["turn_file_id", "frames_in", "turn_voicing", "feature_droppage", "TARGET_DIMS",
           "ProsodySettings", "MeasurementNotAvailable", "measure_turn",
           "compute_turn_prosody", "turn_prosody_from_paths", "file_loudness_offset",
           "droppage_by_dimension"]
