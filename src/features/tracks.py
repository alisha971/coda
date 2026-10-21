"""Frame-level pitch and intensity tracks for one audio file.

A pure function of the audio, the file's own voice-activity segments (part of
the corpus release) and the pinned Praat settings in ``configs/features.yaml``.
It takes no turn boundaries, labels or targets, so nothing downstream can leak
into the measurement. Turn-level target
dimensions are computed from these tracks afterwards, in
``src/features/prosody.py``, which means the audio can be discarded
immediately (process-and-discard) without freezing every target
definition at download time.

Terms:

- **F0** (fundamental frequency): the rate at which the vocal folds vibrate,
  heard as pitch. Measured in Hz; 0 marks a frame where no voicing was found.
- **Pitch strength**: Praat's autocorrelation peak for the chosen F0 candidate,
  between 0 and 1; higher means a clearer periodic signal.
- **Intensity**: Praat's energy contour in dB. The corpus is peak-normalised per
  file, so only differences *within* a file are meaningful; the per-file
  loudness offset in the conditioning set handles the rest.

**Where pitch is tracked.** Over every region in
which *either* the released voice-activity segments *or* the released
transcript places speech, the two merged where they overlap or lie within
``pitch_segment_merge_gap_s``. Both sources are part of the corpus release and
neither is a label or a target, so this remains a pure function of the audio
and its metadata.

The transcript side is not optional detail. The 40-session pilot showed turns
of five seconds and longer receiving almost no pitch, because the
voice-activity detector had missed speech the transcript documented: at the
largest recording site the mean share of word time falling inside detected
speech is 0.73 and the 5th percentile is 0.15. Analysing voice-activity
segments alone therefore left documented speech unmeasured. Taking the union
can only widen the analysed region, never narrow it.

**Why pitch is tracked per speech segment.** Praat treats frames whose
amplitude is below ``silence_threshold`` times the *global* maximum of the
analysed sound as silent. The corpus is peak-normalised per file, so a single
loud transient (a microphone bump, a clipped laugh) sets that maximum and
silences genuine speech across the whole file: on one pilot file, voicing
inside speech fell from 65% to 3.5% for this reason alone. Analysing each of
the speaker's own VAD segments separately makes the maximum local to the
speech being measured, which is how Praat is normally applied to speech units,
and keeps every Praat setting at its default. Each segment is padded by one
pitch-analysis window (three periods of the pitch floor, 40 ms at 75 Hz) so
its edges can be analysed. Frames outside speech are unvoiced by definition.

- **Frame level (dBFS)**: RMS level of each 10 ms frame relative to digital
  full scale, computed directly from the samples. Used by channel QC, not as a
  target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import FeatureConfig

TRACK_KEYS = ("f0_hz", "pitch_strength", "intensity_db", "frame_dbfs")


@dataclass
class Tracks:
    """All tracks for one file. Each track is float32 with its own time origin."""

    step_s: float
    f0_t0: float
    f0_hz: np.ndarray
    pitch_strength: np.ndarray
    int_t0: float
    intensity_db: np.ndarray
    frame_dbfs: np.ndarray  # frames of step_s starting at t = 0
    duration_s: float

    def to_npz_dict(self, prefix: str) -> dict[str, np.ndarray]:
        meta = np.array([self.step_s, self.f0_t0, self.int_t0, self.duration_s], dtype=np.float64)
        return {
            f"{prefix}__meta": meta,
            f"{prefix}__f0_hz": self.f0_hz,
            f"{prefix}__pitch_strength": self.pitch_strength,
            f"{prefix}__intensity_db": self.intensity_db,
            f"{prefix}__frame_dbfs": self.frame_dbfs,
        }

    @classmethod
    def from_npz(cls, npz, prefix: str) -> Tracks:
        step, f0_t0, int_t0, dur = npz[f"{prefix}__meta"].tolist()
        return cls(step, f0_t0, npz[f"{prefix}__f0_hz"], npz[f"{prefix}__pitch_strength"],
                   int_t0, npz[f"{prefix}__intensity_db"], npz[f"{prefix}__frame_dbfs"], dur)


def frame_dbfs(x: np.ndarray, sr: int, step_s: float) -> np.ndarray:
    """RMS level per non-overlapping frame, in dB relative to full scale."""
    n = int(round(sr * step_s))
    m = len(x) // n
    if m == 0:
        return np.zeros(0, dtype=np.float32)
    frames = x[: m * n].reshape(m, n).astype(np.float64)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return (20.0 * np.log10(np.maximum(rms, 1e-10))).astype(np.float32)


def merge_intervals(*interval_sets, gap_s: float = 0.0) -> np.ndarray:
    """Union of interval sets, merged where they overlap or lie within ``gap_s``.

    Returns an ``n x 2`` array of start/end seconds, sorted and non-overlapping.
    """
    parts = [np.asarray(s, dtype=np.float64).reshape(-1, 2)
             for s in interval_sets if s is not None and len(s)]
    if not parts:
        return np.zeros((0, 2), dtype=np.float64)
    allr = np.concatenate(parts)
    allr = allr[np.argsort(allr[:, 0], kind="stable")]
    out = [allr[0].copy()]
    for s, e in allr[1:]:
        if s <= out[-1][1] + gap_s:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append(np.array([s, e], dtype=np.float64))
    return np.asarray(out, dtype=np.float64)


def speech_regions(vad: np.ndarray | None, word_spans: np.ndarray | None,
                   cfg: FeatureConfig | None = None) -> np.ndarray:
    """The regions pitch is analysed over, per the configured source."""
    cfg = cfg or FeatureConfig()
    if cfg.pitch_segment_source == "vad":
        return merge_intervals(vad, gap_s=cfg.pitch_segment_merge_gap_s)
    if cfg.pitch_segment_source == "vad_plus_transcript":
        return merge_intervals(vad, word_spans, gap_s=cfg.pitch_segment_merge_gap_s)
    raise ValueError(f"unknown pitch_segment_source {cfg.pitch_segment_source!r}")


def total_seconds(intervals: np.ndarray) -> float:
    """Time covered by a merged interval set."""
    a = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
    return float(np.clip(a[:, 1] - a[:, 0], 0, None).sum()) if a.size else 0.0


def _pitch(snd, cfg: FeatureConfig):
    return snd.to_pitch_ac(
        time_step=cfg.frame_step_s, pitch_floor=cfg.f0_floor_hz,
        max_number_of_candidates=cfg.max_candidates, very_accurate=cfg.very_accurate,
        silence_threshold=cfg.silence_threshold, voicing_threshold=cfg.voicing_threshold,
        octave_cost=cfg.octave_cost, octave_jump_cost=cfg.octave_jump_cost,
        voiced_unvoiced_cost=cfg.voiced_unvoiced_cost, pitch_ceiling=cfg.f0_ceiling_hz,
    )


def pitch_by_segment(x: np.ndarray, sr: int, segments: np.ndarray,
                     cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    """F0 and strength on a fixed grid (frame k centred at (k + 0.5) * step),
    analysing each speech segment separately. Unvoiced (0) outside segments."""
    import parselmouth

    step = cfg.frame_step_s
    n = int(len(x) / sr / step)
    f0 = np.zeros(n, dtype=np.float32)
    strength = np.zeros(n, dtype=np.float32)
    pad = 3.0 / cfg.f0_floor_hz          # one pitch-analysis window
    for s, e in np.asarray(segments, dtype=np.float64).reshape(-1, 2):
        a, b = max(0, int((s - pad) * sr)), min(len(x), int(np.ceil((e + pad) * sr)))
        if b - a < int(2 * pad * sr):     # too short to hold one analysis window
            continue
        pitch = _pitch(parselmouth.Sound(x[a:b], sampling_frequency=sr), cfg)
        sel = pitch.selected_array
        k = np.floor((pitch.xs() + a / sr) / step).astype(int)
        ok = (k >= 0) & (k < n)
        k, fr, st = k[ok], sel["frequency"][ok], sel["strength"][ok]
        # Where padded segments overlap, keep the more confident estimate.
        better = st > strength[k]
        f0[k[better]] = fr[better]
        strength[k[better]] = st[better]
    return f0, strength


def extract_tracks(x: np.ndarray, sr: int, cfg: FeatureConfig | None = None,
                   speech_segments: np.ndarray | None = None) -> Tracks:
    """Praat pitch and intensity, plus frame levels, for a mono signal.

    ``speech_segments`` (an ``n x 2`` array of start/end seconds, normally the
    speaker's own VAD) switches pitch to per-segment analysis. Without it the
    whole signal is analysed at once, which is only safe when no transient
    dominates the file's maximum; the audio pass always passes segments.
    """
    import parselmouth

    cfg = cfg or FeatureConfig()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("extract_tracks expects a mono signal")
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    if speech_segments is None:
        pitch = _pitch(snd, cfg)
        sel = pitch.selected_array
        f0_t0 = float(pitch.x1)
        f0 = sel["frequency"].astype(np.float32)
        strength = sel["strength"].astype(np.float32)
    else:
        f0_t0 = cfg.frame_step_s / 2
        f0, strength = pitch_by_segment(x, sr, speech_segments, cfg)
    intensity = snd.to_intensity(minimum_pitch=cfg.f0_floor_hz, time_step=cfg.frame_step_s,
                                 subtract_mean=cfg.intensity_subtract_mean)
    return Tracks(
        step_s=cfg.frame_step_s,
        f0_t0=f0_t0, f0_hz=f0, pitch_strength=strength,
        int_t0=float(intensity.x1), intensity_db=intensity.values[0].astype(np.float32),
        frame_dbfs=frame_dbfs(x, sr, cfg.frame_step_s),
        duration_s=len(x) / sr,
    )


__all__ = ["TRACK_KEYS", "Tracks", "frame_dbfs", "merge_intervals", "speech_regions",
           "total_seconds", "pitch_by_segment", "extract_tracks"]
